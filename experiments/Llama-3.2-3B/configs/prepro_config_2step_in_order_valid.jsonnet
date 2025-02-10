local CURRENT_DIR = std.extVar("CURRENT_DIR");
local ROOT_DIR = std.extVar("ROOT");
local DATASET_DIR = "%s/datasets" % ROOT_DIR;

local prepro_train_config = import "prepro_config_1step_train.jsonnet";
local DATASET_DIR_NAME = "preprocess";

prepro_train_config + {
  split:: "valid_2step_in_order",
  corpus_path: "/work/datasets/%s/raw/%s/dataset.jsonl" % [
    DATASET_DIR_NAME,
    self.split,
  ],
  output_dirs: [
    "%s/%s/%s" % [
      DATASET_DIR,
      std.extVar("TAG"),
      self.split,
    ],
  ],
}
