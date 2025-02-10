local SPLITED_FILE_PATH = std.split(std.thisFile, "/");
local THIS_FILE_NAME = SPLITED_FILE_PATH[std.length(SPLITED_FILE_PATH) - 1];
local DATA_TYPE = std.split(THIS_FILE_NAME, "_")[0];
local utils = import "./utils.jsonnet";

local train_config = import "./train_2step_distractor_config.jsonnet";

train_config + {
  seed : utils.datatype_to_seed(DATA_TYPE),
}
