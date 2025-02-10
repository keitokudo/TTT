local CURRENT_DIR = std.extVar("CURRENT_DIR");
local ROOT_DIR = std.extVar("ROOT");
local DATASET_DIR = "%s/datasets" % ROOT_DIR;

local test_data_config = import "./prepro_formula_config_2step_in_order_test.jsonnet";

test_data_config + {
  config_file_path: "%s/data_configs/probing_train_2step_in_order_config.jsonnet" % [
    CURRENT_DIR,
  ],
  output_dir: "%s/%s/raw/probing_train_2step_in_order" % [
    DATASET_DIR,
    std.extVar("TAG"),
  ],
  number_of_data: 10000,
    exclude_dataset_paths: super.exclude_dataset_paths + [
      test_data_config.output_dir,
  ],
}
