local CURRENT_DIR = std.extVar("CURRENT_DIR");
local ROOT_DIR = std.extVar("ROOT");
local DATASET_DIR = "%s/datasets" % ROOT_DIR;

local prepro_valid_config = import "prepro_config_1step_valid.jsonnet";
local DATASET_DIR_NAME = "preprocess";

prepro_valid_config + {
  split:: "test_2step_distractor",
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
  with_few_shot_contexts: true,
}
