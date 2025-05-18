local CURRENT_DIR = std.extVar("CURRENT_DIR");
local ROOT_DIR = std.extVar("ROOT");
local DATASET_DIR = "%s/datasets" % ROOT_DIR;

local prepro_test_config = import "prepro_config_2step_test.jsonnet";
local DATASET_DIR_NAME = "preprocess";

prepro_test_config + {
  preprocessor_name: "LMInstructTuningPreProcessor",
  split:: "test_2step_v2_edit_inconsistent_cot",
  corpus_path: "/work/datasets/%s/raw/%s/merged_edited_EditSecondFormulaNumber_0.jsonl" % [
    DATASET_DIR_NAME,
    "test_2step_v2_edit",
  ],
  output_dirs: [
    "%s/%s/%s" % [
      DATASET_DIR,
      std.extVar("TAG"),
      self.split,
    ],
  ],
  source_key: "base_source",
  target_key: "target",
  shift_size: 38,
}
