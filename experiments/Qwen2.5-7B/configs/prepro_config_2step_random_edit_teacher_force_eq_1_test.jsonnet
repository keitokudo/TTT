local CURRENT_DIR = std.extVar("CURRENT_DIR");
local ROOT_DIR = std.extVar("ROOT");
local DATASET_DIR = "%s/datasets" % ROOT_DIR;

local prepro_test_config = import "prepro_config_2step_v2_edit_test.jsonnet";
local DATASET_DIR_NAME = "preprocess";

prepro_test_config + {
  preprocessor_name: "LMDynamicInterventionPreProcessor",
  split:: "test_2step_random_edit",
  corpus_path: "/work/datasets/%s/raw/%s/merged_edited_Edit2RandomFormula_0.jsonl" % [
    DATASET_DIR_NAME,
    self.split,
  ],
  output_dirs: [
    "%s/%s/%s" % [
      DATASET_DIR,
      std.extVar("TAG"),
      "%s_teacher_force_eq_1" % self.split,
    ],
  ],
  shift_size: 7,
}
