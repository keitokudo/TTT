local CURRENT_DIR = std.extVar("CURRENT_DIR");
local ROOT_DIR = std.extVar("ROOT");
local DATASET_DIR = "%s/datasets" % ROOT_DIR;

local base_prepro_config = import "./prepro_config_base.jsonnet";
local DATASET_DIR_NAME = "preprocess";

base_prepro_config + {
  preprocessor_name: "LMInstructTuningPreProcessor",
  split:: "train_2step_distractor",
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
