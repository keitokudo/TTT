local CONFIG_DIR = "./data_configs";
local ROOT_DIR = std.extVar("ROOT");
local DATSET_DIR = "%s/datasets" % ROOT_DIR;

local utils = import "./utils.jsonnet";
local base_config = import "./decode_test_2step_config.jsonnet";
local model_config = import "./model_specific_config.jsonnet";

local test_data_config = import "./prepro_config_2step_v2_edit_teacher_force_test.jsonnet";

base_config + {
  global_setting: super.global_setting + {
    pl_model_name: "LanguageModelDynamicInterventionPL",
  },

  Logger: super.Logger + {
    version: "%s/%s" % [
      $.global_setting.tag,
      "%s_teacher_force" % test_data_config.split,
    ]
  },
  
  Decoding: super.Decoding + {
    hidden_states_save_dir: "%s/hidden_states_%s" % [
      $.Logger.log_dir,
      "%s_teacher_force_sliding" % test_data_config.split,
    ],
    suffix_match_eval: true,
  },
  
  Datasets: super.Datasets + {
    test_data_file_paths: test_data_config.output_dirs,
  },
  
  PatchConfig: {
    patch_start_position: std.extVar("PATCH_START_POSITION"),
    patch_end_position: std.extVar("PATCH_END_POSITION"),
    patch_start_layer: 12,
    # patch_end_layer: 0,
  }
}
