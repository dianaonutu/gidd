import functools
import hashlib
import json
import os
import shutil
from typing import Callable

import numpy as np
import torch
import hydra
from transformers import BatchEncoding, PreTrainedTokenizer
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler



class DistributedEvalSampler(Sampler):
    """
    Taken from: https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/master/src/data/sampler.py

    DistributedEvalSampler is different from DistributedSampler.
    It does NOT add extra samples to make it evenly divisible.
    DistributedEvalSampler should NOT be used for training. The distributed processes could hang forever.
    See this issue for details: https://github.com/pytorch/pytorch/issues/22584
    shuffle is disabled by default

    DistributedEvalSampler is for evaluation purpose where synchronization does not happen every epoch.
    Synchronization should be done outside the dataloader loop.

    Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.

    .. warning::
        In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        # self.total_size = self.num_samples * self.num_replicas
        self.total_size = len(self.dataset)         # true value without extra samples
        indices = list(range(self.total_size))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)             # true value without extra samples

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): _epoch number.
        """
        self.epoch = epoch


def get_dataset(config, num_proc=32):
    test_size = int(config.data.test_size)
    n_proc = min(os.cpu_count(), num_proc)
    train_ds = load_dataset(
        config.data.dataset_name,
        config.data.dataset_subset,
        split=f"train[:-{test_size}]",
        trust_remote_code=config.data.trust_remote_code,
        num_proc=n_proc,
    )
    test_ds = load_dataset(
        config.data.dataset_name,
        config.data.dataset_subset,
        split=f"train[-{test_size}:]",
        trust_remote_code=config.data.trust_remote_code,
        num_proc=n_proc,
    )

    return train_ds, test_ds


def cached_dataset(cache_dir: str, file_name: str, generate_fn: Callable[[], Dataset]) -> Dataset:
    if cache_dir is None:
        return generate_fn()

    cache_path = os.path.join(cache_dir, file_name)
    if os.path.exists(cache_path):
        ds = Dataset.load_from_disk(cache_path)
        return ds
    else:
        ds = generate_fn()
        os.makedirs(cache_dir, exist_ok=True)
        try:
            ds.save_to_disk(cache_path)
        except Exception as e:
            shutil.rmtree(cache_path)
            raise e
        return ds


def tokenize_dataset(
    ds: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int = 512,
    sequence_packing: bool = False,
    batch_size: int = 1024,
    num_proc: int = 32,
):
    n_proc = min(os.cpu_count(), num_proc)
    bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    eos_token_id = tokenizer.eos_token_id or tokenizer.sep_token_id

    tokenizer_max_len = tokenizer.model_max_length
    tokenizer.model_max_length = 10_000_000

    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=False,
            padding=False,
        )["input_ids"]
        tokens = [[bos_token_id] + x + ([] if sequence_packing else [eos_token_id]) for x in tokens]
        if sequence_packing:
            tokens = np.concatenate(tokens, axis=0)
            tokens = tokens[: len(tokens) - len(tokens) % max_seq_len]
            tokens = tokens.reshape(-1, max_seq_len)
        else:
            tokens = [
                np.pad(x, (0, max_seq_len - len(x) % max_seq_len), mode="constant", constant_values=tokenizer.pad_token_id)
                for x in tokens
            ]
            tokens = [x.reshape(-1, max_seq_len) for x in tokens]
            tokens = np.concatenate(tokens, axis=0)
        return {"input_ids": tokens}

    ds = ds.map(
        tokenize_fn,
        batched=True,
        batch_size=batch_size,
        remove_columns=["text"],
        num_proc=n_proc,
    )

    tokenizer.model_max_length = tokenizer_max_len
    return ds


def default_collator(config, tokenizer, examples, text_key="text"):
    examples = [x[text_key] for x in examples]
    return tokenizer(examples, padding="max_length", truncation=True, max_length=config.model.max_seq_len, return_tensors="pt")


def pretokenized_collator(examples, pad_token_id=0, tokens_key="input_ids"):
    input_ids = np.stack([np.array(x[tokens_key]) for x in examples], axis=0)
    attn_masks = (input_ids != pad_token_id).astype(np.int32)
    input_ids = torch.from_numpy(input_ids).to(torch.long)
    attn_masks = torch.from_numpy(attn_masks).to(torch.long)
    return BatchEncoding({"input_ids": input_ids, "attention_mask": attn_masks}, tensor_type="pt", n_sequences=len(input_ids))


def subsample_collator(config, tokenizer, examples, text_key="text"):
    bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    eos_token_id = tokenizer.eos_token_id or tokenizer.sep_token_id

    examples = [x[text_key] for x in examples]
    tokens = tokenizer(examples, truncation=False, return_tensors="np")
    max_length = config.model.max_seq_len
    input_ids = []
    attn_masks = []
    for i in range(len(examples)):
        toks = tokens["input_ids"][i]
        attn_mask = tokens["attention_mask"][i]
        if toks[0] != bos_token_id:
            toks = np.concatenate([[bos_token_id], toks])
            attn_mask = np.concatenate([[1], attn_mask])
        if toks[-1] != eos_token_id:
            toks = np.concatenate([toks, [eos_token_id]])
            attn_mask = np.concatenate([attn_mask, [1]])

        if len(toks) > max_length:
            overflow = len(toks) - max_length
            start_idx = np.random.randint(0, overflow + config.data.max_add_padding)
            toks = toks[start_idx : start_idx + max_length]
            attn_mask = attn_mask[start_idx : start_idx + max_length]
        if len(toks) < max_length:
            underflow = max_length - len(toks)
            toks = np.pad(toks, (0, underflow), mode="constant", constant_values=tokenizer.pad_token_id)
            attn_mask = np.pad(attn_mask, (0, underflow), mode="constant", constant_values=0)
        assert len(toks) == max_length
        assert len(attn_mask) == max_length
        input_ids.append(toks)
        attn_masks.append(attn_mask)
    input_ids = torch.from_numpy(np.array(input_ids)).to(torch.long)
    attn_masks = torch.from_numpy(np.array(attn_masks)).to(torch.long)
    return BatchEncoding({"input_ids": input_ids, "attention_mask": attn_masks}, tensor_type="pt", n_sequences=len(input_ids))


def _get_dataloader(config, ds, shuffle, drop_last, batch_size, collate_fn, eval=False):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if not eval:
            sampler = DistributedSampler(ds, seed=config.training.seed, shuffle=shuffle)
        else:
            sampler = DistributedEvalSampler(ds, seed=config.training.seed, shuffle=shuffle, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        _shuffle = False
    else:
        sampler = None
        _shuffle = shuffle

    return DataLoader(
        ds,
        collate_fn=collate_fn,
        batch_size=batch_size,
        drop_last=drop_last,
        sampler=sampler,
        num_workers=config.data.num_workers,
        shuffle=_shuffle,
        pin_memory=True,
        persistent_workers=True,
    )


def get_dataloaders(config, tokenizer, train_batch_size=None, eval_batch_size=None):
    if train_batch_size is None:
        train_batch_size = config.training.train_batch_size
    if eval_batch_size is None:
        eval_batch_size = config.training.eval_batch_size

    train_ds, test_ds = get_dataset(config)

    if config.data.pre_tokenize:
        max_seq_len = config.model.max_seq_len
        sequence_packing = config.data.sequence_packing
        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "dataset_name": config.data.dataset_name,
                    "subset": config.data.dataset_subset,
                    "tokenizer_name": config.data.tokenizer_name,
                    "max_seq_len": max_seq_len,
                    "sequence_packing": sequence_packing,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        train_ds = cached_dataset(
            cache_dir=hydra.utils.to_absolute_path(config.data.cache_dir),
            file_name=f"cache-{config.data.dataset_name.replace('/', '--')}-train-{cache_key}",
            generate_fn=functools.partial(tokenize_dataset, ds=train_ds, tokenizer=tokenizer, max_seq_len=max_seq_len, sequence_packing=sequence_packing),
        )
        test_ds = cached_dataset(
            cache_dir=hydra.utils.to_absolute_path(config.data.cache_dir),
            file_name=f"cache-{config.data.dataset_name.replace('/', '--')}-test-{cache_key}",
            generate_fn=functools.partial(tokenize_dataset, ds=test_ds, tokenizer=tokenizer, max_seq_len=max_seq_len, sequence_packing=sequence_packing),
        )

        collate_fn = functools.partial(pretokenized_collator, pad_token_id=tokenizer.pad_token_id, tokens_key="input_ids")
    else:
        if config.data.sequence_packing:
            raise ValueError("Sequence packing requires pre-tokenization.")

        collate_fn = functools.partial(subsample_collator, config, tokenizer, text_key="text")

    train_dl = _get_dataloader(config, train_ds, shuffle=True, drop_last=True, batch_size=train_batch_size, collate_fn=collate_fn)
    test_dl = _get_dataloader(config, test_ds, shuffle=False, drop_last=True, batch_size=eval_batch_size, collate_fn=collate_fn)

    return train_dl, test_dl