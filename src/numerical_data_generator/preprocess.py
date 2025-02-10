import argparse
from pathlib import Path
from copy import deepcopy
from .edit_function_tools import is_num


__all__ = ["biased_answer", "no_preprocess"]




def no_preprocess(operator_config, assignment_configs, numerical_data_generator):
    return operator_config, assignment_configs


def biased_answer(operator_config, assignment_configs, numerical_data_generator):

    random_module = numerical_data_generator.random_module
    
    if random_module.random() < 0.5:
        return operator_config, assignment_configs
    
    assert numerical_data_generator.dtype == "int", "Only support indeger"
    
    operator_config = deepcopy(operator_config)
    assignment_configs = deepcopy(assignment_configs)

    pqa_triple_list = numerical_data_generator.get_pqa_triple_from_configs(
        operator_config,
        assignment_configs,
        separate=True
    )

    assert len(pqa_triple_list) == 1, "Only support \"ask_last_question\" mode"
    passage, question, answer = pqa_triple_list[0]
    answer = int(answer)
    
    new_answer = (answer + 5) - ((answer + 5) % 10)
    
    assert new_answer % 10 == 0, "Method for make new answer is false..."
    last_conf = assignment_configs[-1]
    assert len(last_conf["format"]) == 2, "The length of last_conf[\"format\"] is expected 2" 
    
    nums_list = list(
        filter(
            lambda x: is_num(x, numerical_data_generator.dtype),
            last_conf["format"]
        )
    )
    assert len(nums_list) == 1, "Expect only 1 number"
    last_conf_num = int(nums_list[0])
    
    if last_conf["type"] == "Sub":
        if is_num(last_conf["format"][-1], numerical_data_generator.dtype):
            variable_value = answer + last_conf_num
            new_last_conf_num = variable_value - new_answer
        elif is_num(last_conf["format"][0], numerical_data_generator.dtype):
            variable_value = last_conf_num - answer
            new_last_conf_num = variable_value + new_answer
        
    elif last_conf["type"] == "Add":
        variable_value = answer - last_conf_num
        new_last_conf_num = new_answer - variable_value
    else:
        raise RuntimeError("\"change_answer\" only support \"Sub\" and \"Add\"")
        
    for i, v in enumerate(last_conf["format"]):
        if is_num(v, numerical_data_generator.dtype):
            last_conf["format"][i] = str(new_last_conf_num)

    return operator_config, assignment_configs
