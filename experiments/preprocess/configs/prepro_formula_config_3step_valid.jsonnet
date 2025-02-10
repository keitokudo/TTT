local CURRENT_DIR = std.extVar("CURRENT_DIR");
local ROOT_DIR = std.extVar("ROOT");
local DATASET_DIR = "%s/datasets" % ROOT_DIR;

local train_data_config = import "./prepro_formula_config_3step_train.jsonnet";

train_data_config + {
  config_file_path: "%s/data_configs/valid_3step_config.jsonnet" % [
    CURRENT_DIR,
  ],
  output_dir: "%s/%s/raw/valid_3step" % [
    DATASET_DIR,
    std.extVar("TAG"),
  ],
  number_of_data: 2000,
  exclude_dataset_paths: [
    train_data_config.output_dir,
  ],
}
