local SPLITED_FILE_PATH = std.split(std.thisFile, "/");
local THIS_FILE_NAME = SPLITED_FILE_PATH[std.length(SPLITED_FILE_PATH) - 1];
local DATA_TYPE = std.split(THIS_FILE_NAME, "_")[0];
local utils = import "./utils.jsonnet";

local valid_config = import "./valid_3step_config.jsonnet";

valid_config + {
  seed : utils.datatype_to_seed(DATA_TYPE),
}
