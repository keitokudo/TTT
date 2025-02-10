local SPLITED_FILE_PATH = std.split(std.thisFile, "/");
local THIS_FILE_NAME = SPLITED_FILE_PATH[std.length(SPLITED_FILE_PATH) - 1];
local DATA_TYPE = std.split(THIS_FILE_NAME, "_")[0];
local utils = import "./utils.jsonnet";
local MAX_NUM_FORMULAS = 3;
local MIN_NUM_FORMULAS = 3;

local PREMITIVE_CONFIG_LIST = import "./premitive_configs.jsonnet";

local filter_available_configs(i, premitive_configs) = std.filter(
  function(c) c.num_variables <= i,
  premitive_configs
);

local apply_args_index(i, conf) = 
  if conf.num_variables == 0 then
    conf.generate_config()
  else if conf.num_variables == 1 then
    conf.generate_config(i - 1)
  else if conf.num_variables == 2 then
    conf.generate_config(i - 1, i - 2)
  else
    null;

local merge(contexts, next_configs) =
  [
    context + [conf]
    for context in contexts
    for conf in next_configs
  ];

local get_last_element(l) = l[std.length(l) - 1];


local get_new_configs(i, primitive_configs) =
  [
    apply_args_index(i, conf)
    for conf in filter_available_configs(i, primitive_configs)
  ];

local get_contexts(i, primitive_configs) =
 
  if i == -1 then
    [[]]
  else
    local contexts = get_contexts(i - 1, primitive_configs);
    local new_configs = get_new_configs(i, primitive_configs);
    merge(contexts, new_configs);


local formula_configs = std.flattenArrays(
  [
    get_contexts(i, PREMITIVE_CONFIG_LIST)
    for i in std.reverse(std.range(MIN_NUM_FORMULAS - 1, MAX_NUM_FORMULAS - 1))
  ]
);

{
  seed : utils.datatype_to_seed(DATA_TYPE),
  symbol_selection_slice: "0:26",
  max_number_of_question : "inf",
  
  min_value: 0,
  max_value: 9,
  
  dtype : "int",
  shuffle_order: true,
  # output_type : "ask_last_question",
  output_type : "ask_all_variables",
  
  with_intermediate_reasoning_steps: true,
  intermediate_reasoning_step_type: "simple_backtracking",
  
  generation_rules : [
    {
      type : "template",
      selection_probability : 1.0,
      
      assignment_format : conf,
      
      operator : {
	type : ["Check"],
	selection_probabilities : [1.0], 
	format : [[i] for i in std.range(0, std.length(conf) - 1)]
      },
    } for conf in formula_configs
  ],
  
  constraints: {
    NumberRangeConstraint: {
      passage_range: [$.min_value, $.max_value],
      answer_range: self.passage_range,
      scratchpad_range: self.passage_range,
    },
  },
}
