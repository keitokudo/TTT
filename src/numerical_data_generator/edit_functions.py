from pprint import pprint
from more_itertools import ilen
from copy import deepcopy

from .edit_function_tools import fill_template, apply_commutative, is_num

__all__ = ["add_step", "add_dummy", "change_value", "change_variable", "change_variable_order", "remove_last_step", "change_digit", "chanage_operator", "change_formula_order", "add_step_from_midle_step"]


def add_step(operator_config, assignment_configs, rule_dict, numerical_data_generator, random_module):
    """
    ステップ数を1段追加する
    - 利用するoperatorのformatは"var"を1つしか利用できない (1つは利用しなけらばならない). 
    - operator_configの計算で利用されている変数は1つ
    """
    assert type(rule_dict.get("operators")) is list, "You have to define \"operators\" attribute!"
    operator_config = deepcopy(operator_config)
    assignment_configs = deepcopy(assignment_configs)

    symbols_set = set(numerical_data_generator.symbols)
    used_symbos_set = set(config["variable"] for config in assignment_configs)
    candidate_symbols = tuple(symbols_set - used_symbos_set)
    assert len(candidate_symbols) >= 1, "No more symbols to use an addtional equation!"


    # operator_configの計算で利用されている変数は1種類
    available_variables = list(set(filter(lambda s: not is_num(s, numerical_data_generator.dtype), operator_config["format"])))
    assert len(available_variables) == 1, "The number of variable in operator_config[\"format\"] must be 1!"
    
    
    candidates = []
    for operator_dict in rule_dict["operators"]:
        operator_type = operator_dict["type"]
        operator_format = list(map(lambda x: tuple(x) if type(x) is list else x, operator_dict["format"]))

        if operator_dict.get("commutative", False):
            operator_format = apply_commutative(operator_format)

        acceptable_format = []

        # operatorとして利用されている変数が1つか確認しつつ, operatorの候補を取得する. 
        for of in operator_format:
            if of == "var":
                candidates.append({"type": operator_type, "format": of})
            elif type(of) is tuple:
                required_var_count = ilen(filter(lambda x: x == "var", of))
                if required_var_count == 1:
                    candidates.append({"type": operator_type, "format":of})
                else:
                    raise Exception("The nuber of \"var\" you can use as format is 1!")
            else:
                raise Exception("The nuber of \"var\" you can use as format is 1!")

    
    selected_oerpator_dict = random_module.choice(candidates)
    new_operator_config = fill_template(selected_oerpator_dict, candidate_symbols, available_variables, numerical_data_generator, random_module, use_index=False)
    
    new_operator_config["reasning_index"] = assignment_configs[-1]["reasning_index"] + 1


    if numerical_data_generator.shuffle_order:
        repr_insert_index = random_module.randint(0, new_operator_config["reasning_index"])
        new_operator_config["repr_index"] = repr_insert_index
        for conf in assignment_configs:
            if conf["repr_index"] >= repr_insert_index:
                conf["repr_index"] += 1
    else:
        new_operator_config["repr_index"] = new_operator_config["reasning_index"]
                
    
    assignment_configs.append(new_operator_config)
    
    
    # operator_configで利用する変数を書き換える
    operator_config["format"] = [new_operator_config["variable"] if c == available_variables[0] else c for c in operator_config["format"]]

    return operator_config, assignment_configs

    


def add_dummy(operator_config, assignment_configs, rule_dict, numerical_data_generator, random_module):
    """
    推論経路に関係のない, ダミーの計算(式)を追加する. 
    ダミーの式を追加する位置はランダムに決定される. 
    """
    

    operator_config = deepcopy(operator_config)
    assignment_configs = deepcopy(assignment_configs)

    assert type(rule_dict.get("operators")) is list, "You have to define \"operators\" attribute!"

    symbols_set = set(numerical_data_generator.symbols)
    used_symbos_set = set(config["variable"] for config in assignment_configs)
    candidate_symbols = tuple(symbols_set - used_symbos_set)
    assert len(candidate_symbols) >= 1, "No more symbols to use an addtional equation!"
    
    # insert_indexの前に挿入する

    for _ in range(10):
        insert_index = random_module.choice(list(range(len(assignment_configs))))
        history_configs = assignment_configs[:insert_index]
        available_variables = [c["variable"] for c in history_configs if c["variable"] is not None]
        candidates = []

        for operator_dict in rule_dict["operators"]:
            operator_type = operator_dict["type"]
            operator_format = list(map(lambda x: tuple(x) if type(x) is list else x, operator_dict["format"]))

            if operator_dict.get("commutative", False):
                operator_format = apply_commutative(operator_format)
                
            acceptable_format = []
            for of in operator_format:

                if of == "num":
                    candidates.append({"type": operator_type, "format": of})
                elif of == "var" and len(available_variables) >= 1:
                    candidates.append({"type": operator_type, "format": of})
                elif type(of) is tuple:
                    required_var_count = ilen(filter(lambda x: x == "var", of))
                    if len(available_variables) >= required_var_count:
                        candidates.append({"type": operator_type, "format":of})

        if len(candidates):
            break

    else:
        raise RuntimeError("No candidates were generated...")
            
            
    selected_oerpator_dict = random_module.choice(candidates)
    new_operator_config = fill_template(selected_oerpator_dict, candidate_symbols, available_variables, numerical_data_generator, random_module, use_index=False)


    new_operator_config["reasning_index"] = insert_index
    

    if numerical_data_generator.shuffle_order:
        repr_insert_index = random_module.randint(0, assignment_configs[-1]["reasning_index"] + 1)
        new_operator_config["repr_index"] = repr_insert_index
        for conf in assignment_configs:
            if conf["repr_index"] >= repr_insert_index:
                conf["repr_index"] += 1
            if conf["reasning_index"] >= new_operator_config["reasning_index"]:
                conf["reasning_index"] += 1

                
    else:
        new_operator_config["repr_index"] = new_operator_config["reasning_index"]
        for conf in assignment_configs:
            if conf["repr_index"] >= new_operator_config["repr_index"]:
                conf["repr_index"] += 1
            if conf["reasning_index"] >= new_operator_config["reasning_index"]:
                conf["reasning_index"] += 1

    
    assignment_configs.insert(insert_index, new_operator_config)
    return operator_config, assignment_configs


    


def change_value(operator_config, assignment_configs, rule_dict, numerical_data_generator, random_module):
    operator_config = deepcopy(operator_config)
    assignment_configs = deepcopy(assignment_configs)
    
    num_count = 0
    
    for config in assignment_configs + [operator_config]:
        config_format = config["format"]
        if type(config_format) is list:
            num_count += ilen(filter(lambda x: is_num(x, numerical_data_generator.dtype), config_format))
        elif (type(config_format) is str) and is_num(config_format, numerical_data_generator.dtype):
            num_count += 1
        else:
            raise NotImplementedError()

    assert num_count > 0, "There are no number in this equation"

    selected_num_index = random_module.choice(tuple(range(1, num_count + 1)))
    
    num_count = 0
    for j, config in enumerate(assignment_configs):
        config_format = config["format"]
        if type(config_format) is list:
            for i, s in enumerate(config_format):
                if is_num(s, numerical_data_generator.dtype):
                    num_count += 1
                    if num_count == selected_num_index:
                        config_format[i] = str(
                            numerical_data_generator.random_func(
                                numerical_data_generator.min_value,
                                numerical_data_generator.max_value
                            )
                        )
                        break
            else:
                continue
            break
            
        elif (type(config_format) is str) and is_num(config_format, numerical_data_generator.dtype):
            num_count += 1
            if num_count == selected_num_index:
                assignment_configs[j]["format"] = str(
                    numerical_data_generator.random_func(
                        numerical_data_generator.min_value,
                        numerical_data_generator.max_value
                    )
                )
                break
        else:
            raise NotImplementedError()
                
    else:
        config_format = operator_config["format"]
        if type(config_format) is list:
            for i, s in enumerate(config_format):
                if is_num(s, numerical_data_generator.dtype):
                    num_count += 1
                    if num_count == selected_num_index:
                        config_format[i] = str(
                            numerical_data_generator.random_func(
                                numerical_data_generator.min_value,
                                numerical_data_generator.max_value
                            )
                        )
                        break

        elif (type(config_format) is str) and is_num(config_format, numerical_data_generator.dtype):
            num_count += 1
            if num_count == selected_num_index:
                assignment_configs[j]["format"] = str(
                    numerical_data_generator.random_func(
                        numerical_data_generator.min_value,
                        numerical_data_generator.max_value
                    )
                )
        else:
            raise NotImplementedError()
        
    assert num_count == selected_num_index, "Mismatch was occured..."
    
    return operator_config, assignment_configs




def change_variable(operator_config, assignment_configs, rule_dict, numerical_data_generator, random_module):
    """
    利用している変数名を1つ変える
    """
    operator_config = deepcopy(operator_config)
    assignment_configs = deepcopy(assignment_configs)
    
    symbols_set = set(numerical_data_generator.symbols)
    used_symbos_set = set(config["variable"] for config in assignment_configs)
    candidate_symbols = tuple(symbols_set - used_symbos_set)
    
    original_symbol = random_module.choice(tuple(used_symbos_set))
    new_symbol = random_module.choice(candidate_symbols)

    for config in assignment_configs:
        if config["variable"] == original_symbol:
            config["variable"] = new_symbol

        config["format"] = [new_symbol if s == original_symbol else s for s in config["format"]]
    operator_config["format"] = [new_symbol if s == original_symbol else s for s in operator_config["format"]]
    
    return operator_config, assignment_configs




def change_variable_order(operator_config, assignment_configs, rule_dict, numerical_data_generator, random_module):
    """
    利用している変数名の登場順序を変更する. 
    変数を2つ以上利用していないと入れ替えられないので, この関数は利用できない
    """
    operator_config = deepcopy(operator_config)
    assignment_configs = deepcopy(assignment_configs)

    available_variables = [c["variable"] for c in assignment_configs if c["variable"] is not None]
    assert len(available_variables) >= 2, "This edit_function can be used when numbaer of variable used in equations is more than 2!"
    
    
    v1, v2 = random_module.sample(available_variables, 2)
    
    for config in assignment_configs + [operator_config]:
        for i, s in enumerate(config["format"]):
            if s == v1:
                config["format"][i] = v2
            elif s == v2:
                config["format"][i] = v1
                
        if config.get("variable", "") == v1:
            config["variable"] = v2
            continue
        elif config.get("variable", "") == v2:
            config["variable"] = v1

    
    return operator_config, assignment_configs




def remove_last_step(operator_config, assignment_configs, rule_dict, numerical_data_generator, random_module):
    """
    最後の計算推論ステップを削除して, 推論ステップ数を1減らす. 
    - operator_configの計算内容と, この結果を導くのに必要な変数を導出する計算は1種類の変数に依存している必要があります
    """
    operator_config = deepcopy(operator_config)
    assignment_configs = deepcopy(assignment_configs)
    
    deleted_varialble = list(set(filter(lambda x: not is_num(x, numerical_data_generator.dtype), operator_config["format"])))
    assert len(deleted_varialble) == 1, "Number of variable in operator_config[\"format\"] must be 1!"
    deleted_varialble = deleted_varialble[0]
    
    deleted_config = None
    #delete_index = None
    for config in reversed(assignment_configs):
        if config["variable"] == deleted_varialble:
            deleted_config = config
            #delete_index = i
    assert deleted_config is not None, f"Maybe, \"{deleted_varialble}\" is not defined"

    
    
    used_variable_in_deleted_config = list(set(filter(lambda x: not is_num(x, numerical_data_generator.dtype), deleted_config["format"])))
    assert len(used_variable_in_deleted_config) == 1, "Number of variable in deleted_config[\"format\"] must be 1!"
    used_variable_in_deleted_config = used_variable_in_deleted_config[0]

    # operator_config内の変数を置き換え
    operator_config["format"] = [used_variable_in_deleted_config if s == deleted_varialble else s for s in operator_config["format"]]

    # 後ろから探索してremove
    assignment_configs.reverse()
    assignment_configs.remove(deleted_config)
    assignment_configs.reverse()


    for conf in assignment_configs:
        if conf["repr_index"] > deleted_config["repr_index"]:
            conf["repr_index"] -= 1

    return operator_config, assignment_configs



def change_digit(operator_config, assignment_configs, rule_dict, numerical_data_generator, random_module):
    assert numerical_data_generator.dtype == "int", "Support only \"int\"..."
    operator_config = deepcopy(operator_config)
    assignment_configs = deepcopy(assignment_configs)
    
    num_count = 0
    
    for config in assignment_configs + [operator_config]:
        config_format = config["format"]
        if type(config_format) is list:
            num_count += ilen(filter(lambda x: is_num(x, numerical_data_generator.dtype), config_format))
        elif (type(config_format) is str) and is_num(config_format, numerical_data_generator.dtype):
            num_count += 1
        else:
            raise NotImplementedError()

    assert num_count > 0, "There are no number in this equation"

    selected_num_index = random_module.choice(tuple(range(1, num_count + 1)))

    def convert_digits(num):
        str_num = str(num)
        change_index = random_module.choice(range(len(str_num)))
        
        while True:
            new_digit = str(numerical_data_generator.random_func(0, 9))
            if not str_num[change_index] == new_digit:
                list_num = list(str_num)
                list_num[change_index] = new_digit
                str_num =  "".join(list_num)
                if str_num[0] == "0" and len(str_num) > 1:
                    continue
                break
        return str_num
        
        
    
    num_count = 0
    for j, config in enumerate(assignment_configs):
        config_format = config["format"]
        if type(config_format) is list:
            for i, s in enumerate(config_format):
                if is_num(s, numerical_data_generator.dtype):
                    num_count += 1
                    if num_count == selected_num_index:
                        config_format[i] = convert_digits(s)
                        break
            else:
                continue
            break
            
        elif (type(config_format) is str) and is_num(config_format, numerical_data_generator.dtype):
            num_count += 1
            if num_count == selected_num_index:
                assignment_configs[j]["format"] = convert_digits(s)
                break
        else:
            raise NotImplementedError()
                
    else:
        config_format = operator_config["format"]
        if type(config_format) is list:
            for i, s in enumerate(config_format):
                if is_num(s, numerical_data_generator.dtype):
                    num_count += 1
                    if num_count == selected_num_index:
                        config_format[i] = convert_digits(s)
                        break

        elif (type(config_format) is str) and is_num(config_format, numerical_data_generator.dtype):
            num_count += 1
            if num_count == selected_num_index:
                assignment_configs[j]["format"] = convert_digits(s)
        else:
            raise NotImplementedError()
        
    assert num_count == selected_num_index, "Mismatch was occured..."
    
    return operator_config, assignment_configs
    



def chanage_operator(operator_config, assignment_configs, rule_dict, numerical_data_generator, random_module):
    operator_config = deepcopy(operator_config)
    assignment_configs = deepcopy(assignment_configs)
    
    assert type(rule_dict.get("operator_convert_rules")) is list, "You have to define \"operator_convert_rules\" attribute!"


    operator_config["type"] = operator_config["ope"] 

    combined_configs = assignment_configs + [operator_config]
    befores = [d["before"]  for d in rule_dict["operator_convert_rules"]]
    convert_candidates_index = []

    
    for i, config in enumerate(combined_configs):
        if config["type"] in befores:
            convert_candidates_index.append(i)

    assert len(convert_candidates_index) > 0, "No candidate to convert..."
        
    selected_config_index = random_module.choice(convert_candidates_index)

    old_type = combined_configs[selected_config_index]["type"]
    convert_rule_candidates = list(
        filter(
            lambda d: d["before"] == old_type,
            rule_dict["operator_convert_rules"]
        )
    )
    assert len(convert_rule_candidates)  > 0, "No appropriate rule is exist..."

    selected_rule = random_module.choice(convert_rule_candidates)
    combined_configs[selected_config_index]["type"] = selected_rule["after"]

    operator_config["ope"] = operator_config["type"]
    del operator_config["type"]
    
    return operator_config, assignment_configs





def change_formula_order(operator_config, assignment_configs, rule_dict, numerical_data_generator, random_module):
    
    operator_config = deepcopy(operator_config)
    assignment_configs = deepcopy(assignment_configs)

    new_order_index = random_module.sample(
        range(len(assignment_configs)),
        len(assignment_configs)
    )
    for conf, index in zip(assignment_configs, new_order_index):
        conf["repr_index"] = index

    return operator_config, assignment_configs




def add_step_from_midle_step(operator_config, assignment_configs, rule_dict, numerical_data_generator, random_module):
    operator_config = deepcopy(operator_config)
    assignment_configs = deepcopy(assignment_configs)

    assert operator_config["ope"] == "Check", "Only support \"Check\" for assignment_configs[\"ope\"]"

    check_variable = operator_config["format"][0]
    check_target_conf = list(filter(lambda d: d["variable"] == check_variable, assignment_configs))
    
    assert len(check_target_conf) == 1, "There are more than 2 configs as target"
    check_target_conf = check_target_conf[0]

    new_check_target_index = check_target_conf["reasning_index"] + 1
    assert new_check_target_index < len(assignment_configs), "\"new_check_target_index\" is out of range..."

    operator_config["format"] = [assignment_configs[new_check_target_index]["variable"]]
    
    return operator_config, assignment_configs



"""
def add_step_with_distractor(operator_config, assignment_configs, rule_dict, numerical_data_generator, random_module):
    
    # ステップ数を1段追加する
    # multi_step_calc_inferences_with_midle_stepの形式
    # A=1, B=2,   C=A+3,D=B+4,  E=D+5,F=C+6,   F?の形式に対応

    # - 利用するoperatorのformatは"var"を1つしか利用できない (1つは利用しなけらばならない). 
    # - operator_configの計算で利用されている変数は1つ
    
    assert type(rule_dict.get("operators")) is list, "You have to define \"operators\" attribute!"
    operator_config = deepcopy(operator_config)
    assignment_configs = deepcopy(assignment_configs)

    symbols_set = set(numerical_data_generator.symbols)
    used_symbos_set = set(config["variable"] for config in assignment_configs)
    candidate_symbols = tuple(symbols_set - used_symbos_set)
    assert len(candidate_symbols) >= 1, "No more symbols to use an addtional equation!"
















    

    # operator_configの計算で利用されている変数は1種類
    available_variables = list(set(filter(lambda s: not is_num(s, numerical_data_generator.dtype), operator_config["format"])))
    assert len(available_variables) == 1, "The number of variable in operator_config[\"format\"] must be 1!"
    
    
    candidates = []
    for operator_dict in rule_dict["operators"]:
        operator_type = operator_dict["type"]
        operator_format = list(map(lambda x: tuple(x) if type(x) is list else x, operator_dict["format"]))

        if operator_dict.get("commutative", False):
            operator_format = apply_commutative(operator_format)

        acceptable_format = []

        # operatorとして利用されている変数が1つか確認しつつ, operatorの候補を取得する. 
        for of in operator_format:
            if of == "var":
                candidates.append({"type": operator_type, "format": of})
            elif type(of) is tuple:
                required_var_count = ilen(filter(lambda x: x == "var", of))
                if required_var_count == 1:
                    candidates.append({"type": operator_type, "format":of})
                else:
                    raise Exception("The nuber of \"var\" you can use as format is 1!")
            else:
                raise Exception("The nuber of \"var\" you can use as format is 1!")

    
    selected_oerpator_dict = random_module.choice(candidates)
    new_operator_config = fill_template(selected_oerpator_dict, candidate_symbols, available_variables, numerical_data_generator, random_module, use_index=False)
    
    new_operator_config["reasning_index"] = assignment_configs[-1]["reasning_index"] + 1


    if numerical_data_generator.shuffle_order:
        repr_insert_index = random_module.randint(0, new_operator_config["reasning_index"])
        new_operator_config["repr_index"] = repr_insert_index
        for conf in assignment_configs:
            if conf["repr_index"] >= repr_insert_index:
                conf["repr_index"] += 1
    else:
        new_operator_config["repr_index"] = new_operator_config["reasning_index"]
                
    
    assignment_configs.append(new_operator_config)
    
    
    # operator_configで利用する変数を書き換える
    operator_config["format"] = [new_operator_config["variable"] if c == available_variables[0] else c for c in operator_config["format"]]

    return operator_config, assignment_configs
"""

def Pseudolize(operator_config, assignment_configs, rule_dict, numerical_data_generator, random_module):
    raise NotImplementedError()

