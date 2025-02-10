local CONFIG_DIR = "./data_configs";
local ROOT_DIR = std.extVar("ROOT");
local DATSET_DIR = "%s/datasets" % ROOT_DIR;

local utils = import "./utils.jsonnet";
local base_config = import "./base_config.jsonnet";
local model_config = import "./model_specific_config.jsonnet";

local train_data_config = import "./prepro_config_2step_distractor_train.jsonnet";
local valid_data_config = import "./prepro_config_2step_distractor_valid.jsonnet";
local test_data_config = import "./prepro_config_2step_distractor_test.jsonnet";

base_config + {
  Logger: super.Logger + {
    version: "%s/%s" % [
      $.global_setting.tag,
      test_data_config.split,
    ]
  },
  
  Decoding: super.Decoding + {
    hidden_states_save_dir: "%s/hidden_states_%s" % [
      $.Logger.log_dir,
      test_data_config.split,
    ]
  },
  
  Datasets: super.Datasets + {
    train_data_file_paths: train_data_config.output_dirs,
    valid_data_file_paths: valid_data_config.output_dirs,
    test_data_file_paths: test_data_config.output_dirs,
  },
}
