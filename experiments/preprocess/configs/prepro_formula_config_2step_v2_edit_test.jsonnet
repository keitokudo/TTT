local CURRENT_DIR = std.extVar("CURRENT_DIR");
local ROOT_DIR = std.extVar("ROOT");
local DATASET_DIR = "%s/datasets" % ROOT_DIR;

local train_data_config = import "./prepro_formula_config_2step_train.jsonnet";
local valid_data_config = import "./prepro_formula_config_2step_valid.jsonnet";
local test_data_config = import "./prepro_formula_config_2step_test.jsonnet";

test_data_config + {
  config_file_path: "%s/data_configs/test_2step_v2_edit_config.jsonnet" % [
    CURRENT_DIR,
  ],
  output_dir: "%s/%s/raw/test_2step_v2_edit" % [
    DATASET_DIR,
    std.extVar("TAG"),
  ],
  exclude_dataset_paths: super.exclude_dataset_paths + [
    valid_data_config.output_dir,
  ],
}
