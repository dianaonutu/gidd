# Updated Configs

- `gidd/configs/data/defaults.yaml`:  
  - `cache_dir`  
  - `tokenizer` name (set to path instead of downloading)
  
- `gidd/configs/data/fineweb.yaml`:  
  - `dataset_name`
  
- `gidd/configs/logging/default.yaml`:  
  - `wandb_entity`  
  - `wandb_project`
  
- `gidd/configs/gidd.yaml`:  
  - `data`  
  - `model`  
  - `num_train_steps`
  
- `gidd/configs/mdlm.yaml`:  
  - `data`  
  - `model`  
  - `num_train_steps`
  
- `pretrain_job_script.job`:  
  - `nnodes`  
  - `ngpus`  
  - `time`  
  - `config-name` (which model to use)  
  - `logging.run_name`
