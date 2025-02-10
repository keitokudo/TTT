local CURRENT_DIR = std.extVar("CURRENT_DIR");
local ROOT_DIR = std.extVar("ROOT");
local DATASET_DIR = "%s/datasets" % ROOT_DIR;
local model_config = import "./model_specific_config.jsonnet";

{
  seed: 42,
  test_set: true, # To preserve the order of the samples
  tokenizer_name_or_path: model_config.model_name_or_path,
  model_max_length: model_config.model_max_length,
}
