local SPLITED_FILE_PATH = std.split(std.thisFile, "/");
local THIS_FILE_NAME = SPLITED_FILE_PATH[std.length(SPLITED_FILE_PATH) - 1];
local DATA_TYPE = std.split(THIS_FILE_NAME, "_")[0];
local utils = import "./utils.jsonnet";

{
  seed : utils.datatype_to_seed(DATA_TYPE),
  symbol_selection_slice: "0:26",
  max_number_of_question : "inf",
  
  min_value: 0,
  max_value: 9,
  
  dtype : "int",
  shuffle_order: false,
  reverse_order: true,
  output_type : "ask_all_variables",

  with_intermediate_reasoning_steps: true,
  intermediate_reasoning_step_type: "simple_backtracking",
  
  generation_rules : [
    
    {
      type : "template",
      selection_probability : 1.0,

      assignment_format : [
	
	{
	  type: ["Add", "Sub"],
	  format: [["num", "num"]]
	},
	      
	{
	  type: ["Add", "Sub"],
	  format: [["num", 0]],
	  # commutative: true,
	},

	{
	  type: ["Add", "Sub"],
	  format: [["num", 1]],
	  # commutative: true,
	}

      ],
      
      operator : {
	type : ["Check"],
	selection_probabilities : [1.0], 
	format : [-1]
      }
    },
    
  ],
  
  
  constraints: {
    NumberRangeConstraint: {
      passage_range: [$.min_value, $.max_value],
      answer_range: self.passage_range,
      scratchpad_range: self.passage_range,
    },
  },
}
