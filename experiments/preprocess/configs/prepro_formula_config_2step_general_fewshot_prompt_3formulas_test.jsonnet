local CURRENT_DIR = std.extVar("CURRENT_DIR");
local ROOT_DIR = std.extVar("ROOT");
local DATASET_DIR = "%s/datasets" % ROOT_DIR;

local valid_data_config = import "./prepro_formula_config_2step_valid.jsonnet";
local train_data_config = import "./prepro_formula_config_general_fewshot_prompt_3formulas_train.jsonnet";

valid_data_config + {
  config_file_path: "%s/data_configs/test_2step_distractor_config.jsonnet" % [
    CURRENT_DIR,
  ],
  output_dir: "%s/%s/raw/test_2step_distractor_general_fewshot_prompt_3formulas" % [
    DATASET_DIR,
    std.extVar("TAG"),
  ],
  number_of_data: 2000,
  exclude_dataset_paths: super.exclude_dataset_paths + [
    valid_data_config.output_dir,
  ],
  few_shot_sample_dataset_path: train_data_config.output_dir,
  number_of_few_shot_samples: 50,
}
