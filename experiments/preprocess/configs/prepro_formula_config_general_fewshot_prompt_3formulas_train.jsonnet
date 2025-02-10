local CURRENT_DIR = std.extVar("CURRENT_DIR");
local ROOT_DIR = std.extVar("ROOT");
local DATASET_DIR = "%s/datasets" % ROOT_DIR;

{
  preprocessor_name: "DentakuPreProcessor",
  seed: 42,

  config_file_path: "%s/data_configs/train_general_fewshot_prompt_3formulas_config.jsonnet" % [
    CURRENT_DIR,
  ],
  
  output_dir: "%s/%s/raw/train_general_fewshot_prompt_3formulas" % [
    DATASET_DIR,
    std.extVar("TAG"),
  ],
  number_of_data: 100,
  eos_token: "\n",
  pre_tokenization_method: "char",
}
