local CONFIG_DIR = "./data_configs";
local ROOT_DIR = std.extVar("ROOT");
local DATSET_DIR = "%s/datasets" % ROOT_DIR;

local utils = import "./utils.jsonnet";
local base_config = import "./decode_test_2step_config.jsonnet";
local model_config = import "./model_specific_config.jsonnet";

local test_data_config = import "./prepro_config_2step_v2_edit_test.jsonnet";

base_config + {
  global_setting: super.global_setting + {
    pl_model_name: "LanguageModelDynamicInterventionPL",
  },

  Logger: super.Logger + {
    version: "%s/%s" % [
      $.global_setting.tag,
      test_data_config.split,
    ]
  },
  
  Decoding: super.Decoding + {
    hidden_states_save_dir: "%s/hidden_states_%s" % [
      $.Logger.log_dir,
      test_data_config.split,
    ]
  },
  
  Datasets: super.Datasets + {
    test_data_file_paths: test_data_config.output_dirs,
  },
  
  PatchConfig: {
    patch_start_position: 13,
    patch_end_position: 14,
    patch_start_layer: 12,
  }
}
