import json
import shutil
import random
from pathlib import Path
from dataclasses import asdict, dataclass

import torch
import numpy as np
from transformers import AutoTokenizer
from omegaconf import OmegaConf
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

from gidd.diffusion_process import get_noise_schedule
from gidd.modeling import get_model
from gidd.trainer import DiffusionTrainer, get_trainer
from gidd.loss import get_loss
from gidd.optimizer import get_optimizer


@dataclass
class TrainingState:
    epoch: int = 0
    epoch_start_step: int = 0
    step: int = 0
    total_tokens: int = 0
    total_flops: float = 0.0
    start_time: float = -1
    curr_time: float = -1

def build_checkpoint_state(trainer: DiffusionTrainer, optimizer, state: TrainingState):
    """
    Build in-memory checkpoint state. MUST be called on ALL ranks because it
    may trigger FSDP collectives (e.g., full_state_dict()).
    Returns (model_state_dict, optim_state_dict, extra_state).
    """
    if isinstance(trainer.model, (FSDP, DDP)):
        model_state_dict, optim_state_dict = get_state_dict(trainer.model, optimizer)
    else:
        model_state_dict, optim_state_dict = (
            trainer.model.state_dict(),
            optimizer.state_dict()
        )

    if hasattr(trainer, "noise_schedule"):
        noise_schedule_state_dict = trainer.noise_schedule.state_dict()
    else:
        noise_schedule_state_dict = []
    return model_state_dict, optim_state_dict, noise_schedule_state_dict


def write_checkpoint_to_disk(path: Path, trainer: DiffusionTrainer, model_state_dict, optim_state_dict, noise_schedule_state_dict, state: TrainingState):
    """
    Only rank 0 should call this function (it writes files).
    """
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(exist_ok=True, parents=True)

    # save config
    OmegaConf.save(config=trainer.config, f=path / "config.yaml", resolve=True)

    torch.save(model_state_dict, path / "model.pt")
    torch.save(optim_state_dict, path / "optimizer.pt")
    trainer.tokenizer.save_pretrained(path)

    # save noise schedule
    if noise_schedule_state_dict:
        torch.save(noise_schedule_state_dict, path / "noise_schedule.pt")

    # save training state
    with open(path / "state.json", "w") as f:
        json.dump(asdict(state), f)


def save_checkpoint(path, trainer: DiffusionTrainer, optimizer, state: TrainingState):
    # All ranks build the checkpoint (necessary for FSDP)
    model_state_dict, optim_state_dict, noise_schedule_state_dict = build_checkpoint_state(trainer, optimizer, state)
    dist.barrier()

    if not dist.is_initialized() or dist.get_rank() == 0:
        write_checkpoint_to_disk(path, trainer, model_state_dict, optim_state_dict, noise_schedule_state_dict, state)

    dist.barrier()
    
    # Free big objects on non-main ranks
    if dist.is_initialized() and dist.get_rank() != 0:
        del model_state_dict
        del optim_state_dict
        del noise_schedule_state_dict

    # final barrier optional
    dist.barrier()


def load_checkpoint(path, device=None):
    config = OmegaConf.load(Path(path, "config.yaml"))

    tokenizer = AutoTokenizer.from_pretrained(path)

    model = get_model(config, tokenizer, device="cpu")

    model_state_dict = torch.load(Path(path, "model.pt"), map_location="cpu", weights_only=True)
    if isinstance(model, (FSDP, DDP)):
        set_state_dict(model, model_state_dict)
    else:
        model.load_state_dict(model_state_dict)

    if device is not None:
        model.to(device)

    if config.model.type == "diffusion":
        noise_schedule = get_noise_schedule(config, tokenizer)
        schedule_path = Path(path, "noise_schedule.pt")
        if schedule_path.exists():
            schedule_state_dict = torch.load(schedule_path, map_location="cpu", weights_only=True)
            noise_schedule.load_state_dict(schedule_state_dict)
        if device is not None:
            noise_schedule.to(device)
    else:
        noise_schedule = None
    
    return model, noise_schedule, tokenizer, config


def load_checkpoint_for_training(path, config=None, device=None, dtype=None):
    # load model, noise_schedule, tokenizer and config
    model, noise_schedule, tokenizer, old_config = load_checkpoint(path, device=None)
    if config is None:
        # use the config from the checkpoint if none is provided
        config = old_config
    if device:
        noise_schedule.to(device)
    # initialize trainer
    loss_fn = get_loss(config, tokenizer, noise_schedule)
    trainer = get_trainer(config, model, tokenizer, noise_schedule, loss_fn, dtype=dtype)
    if device:
        trainer.to(device)
    # initialize and load optimizer state
    optimizer = get_optimizer(config, trainer)
    opt_state_dict = torch.load(Path(path, "optimizer.pt"), map_location="cpu", weights_only=True)
    if isinstance(trainer.model, (FSDP, DDP)):
        set_state_dict(trainer.model, opt_state_dict, optimizer=optimizer)
    else:
        optimizer.load_state_dict(opt_state_dict)
    # load training state
    with open(Path(path, "state.json")) as f:
        state = TrainingState(**json.load(f))
    # return everything
    return model, noise_schedule, tokenizer, old_config, trainer, optimizer, state


def save_rng_state(path: Path, rank: int):
    rng_state_dict = {
        'cpu_rng_state': torch.get_rng_state(),
        'gpu_rng_state': torch.cuda.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
        'py_rng_state': random.getstate()
    }
    torch.save(rng_state_dict, Path(path, f'rng_state_{rank}.pt'))


def load_rng_state(path: Path, rank: int):
    torch.cuda.set_device(rank)
    rng_state_dict = torch.load(Path(path, f'rng_state_{rank}.pt'), map_location='cpu', weights_only=False)
    torch.set_rng_state(rng_state_dict['cpu_rng_state'])
    torch.cuda.set_rng_state(rng_state_dict['gpu_rng_state'])
    np.random.set_state(rng_state_dict['numpy_rng_state'])
    random.setstate(rng_state_dict['py_rng_state'])
