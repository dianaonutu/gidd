# Updated configs
`cd gidd/configs/data/defaults`: cache_dir & tokenizer name (path to tokenizer instead of downloading)
`cd gidd/configs/data/fineweb`: dataset_name
`cd gidd/configs/logging/default`: wandb_entity and wandb_project
`cd gidd/configs/gidd`: data, model, num_train_steps
`cd gidd/configs/mdlm`: data, model, num_train_steps
`pretrain_job_script`: nnodes, ngpus, time, config-name (which model to use) and logging.run_name