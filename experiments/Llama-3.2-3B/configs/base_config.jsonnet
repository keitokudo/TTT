local CONFIG_DIR = "./data_configs";
local ROOT_DIR = std.extVar("ROOT");
local DATSET_DIR = "%s/datasets" % ROOT_DIR;

local train_data_config = import "./prepro_config_train.jsonnet";
local valid_data_config = import "./prepro_config_valid.jsonnet";
local model_config = import "./model_specific_config.jsonnet";

{
  train_only: false,
  
  global_setting: {
    pl_model_name: "LanguageModelPL",
    seed: 42,
    tag: "%s_seed_%s" % [
      std.extVar("TAG"),
      std.toString(self.seed),
    ],
    log_model_output_dir: "%s/experiment_results/%s" % [ROOT_DIR, self.tag],
    # torch_compile_mode: "default",
    # “default”, “reduce-overhead” or “max-autotune”
    # load_check_point: "%s/last.ckpt" % $.Callbacks.checkpoint_save_path,
  },
  
  Logger: {
    project_name: "dentaku_probing",
    log_dir: "%s/logs" % $.global_setting.log_model_output_dir,
    version: "%s/test" % [$.global_setting.tag],
  },
  
  Trainer: {
    max_epochs: std.length($.Datasets.train_data_file_paths) * 1,
    val_check_interval: "1.0",
    check_val_every_n_epoch: 1,
    log_every_n_steps: 10,
    default_root_dir: "%s/defaults" % $.global_setting.log_model_output_dir,
    weights_save_path: "%s/weights" % $.global_setting.log_model_output_dir,
    strategy : "auto",
    # accumulate_grad_batches: 8,
    gradient_clip_val: 1.0,
    reload_dataloaders_every_n_epochs: if std.length($.Datasets.train_data_file_paths) == 1 then 0 else 1,
    bf16: true,
    precision_mode: "true",
    # fp16: true,
  },
  
  Callbacks: {
    save_top_k: 1,
    checkpoint_save_path: "%s/checkpoints" % $.global_setting.log_model_output_dir,
    early_stopping_patience: -1,
    async_checkpointing: true,
  },
  
  pl_module_setting: {
    lr_scheduler: "cosine",
    lr: 1e-5,
    num_warmup_steps: 50,
    
    model_name_or_path: model_config.model_name_or_path,
    tokenizer_name_or_path: model_config.tokenizer_name_or_path,
    from_scratch: false,
    
    max_new_tokens: 100,
  },
  
  Datasets: model_config.Datasets + {
    train_data_file_paths: train_data_config.output_dirs,
    valid_data_file_paths: valid_data_config.output_dirs,
    num_workers: 8,
    train_data_shuffle: false,
  },
  
  Decoding: model_config.Decoding + {
    save_hidden_states: true,
    save_hideen_states_bf16: true,
  },  
}
