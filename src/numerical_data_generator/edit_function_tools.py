import argparse
from pathlib import Path
from itertools import permutations

def fill_template(selected_oerpator_dict, candidate_symbols, available_variables, numerical_data_generator, random_module, use_index=False):
    
    selected_format = selected_oerpator_dict["format"]
    
    filled_format = []
    if type(selected_format) is tuple:
        format_for_assertion = tuple(map(lambda x : "var" if type(x) is int else x, selected_format))
        
        operator_acceptable_formats = numerical_data_generator.operators_dict[selected_oerpator_dict["type"]].arg_formats
        
        assert numerical_data_generator.format_assert(format_for_assertion, operator_acceptable_formats), f"\"selected_format[\"type\"]\" is not support \"{format_for_assertion}\"."

        
        for elem in selected_format:
            if elem == "num":
                filled_format.append(str(numerical_data_generator.random_func(numerical_data_generator.min_value, numerical_data_generator.max_value)))
            elif elem == "var":
                filled_format.append(random_module.choice(available_variables))
            elif (type(elem) is int) and use_index:
                assert generation_type=="template", "generation_rules:type = \"random\" can not use index selection."
                # - でリストを参照する際に, available_variables(リスト)の所望の位置にアクセスできるようにする. (available_variablesは長さがその時点まで処理したものしか格納されていないため)
                if elem < 0:
                    elem = assumption_length + elem

                filled_format.append(available_variables[elem])
            else:
                raise NotImplementedError()



            
    else:

        # 代入に用いる, 変数, 数値の形式が正しいか確認
        format_for_assertion = "var" if type(selected_format) is int else selected_format


        operator_acceptable_formats = numerical_data_generator.operators_dict[selected_oerpator_dict["type"]].arg_formats

        assert numerical_data_generator.format_assert((format_for_assertion, ), operator_acceptable_formats), f"\"selected_format[\"type\"]\" is not support \"{format_for_assertion}\"."

        if selected_format == "num":
            filled_format.append(str(numerical_data_generator.random_func(numerical_data_generator.min_value, numerical_data_generator.max_value)))
        elif selected_format == "var":
            filled_format.append(random_module.choice(available_variables))
        elif (type(selected_format) is int) and use_index:
            assert generation_type=="template", "generation_rules:type = \"random\" can not use index selection."
            if elem < 0:
                elem = assumption_length + elem

            filled_format.append(available_variables[elem])
        else:
            raise NotImplementedError()


    return {
        "variable": random_module.choice(candidate_symbols),
        "type": selected_oerpator_dict["type"],
        "format": filled_format
    }
        


def apply_commutative(operator_format):
    temp_operator_formats = []
    for of in operator_format:
        if type(of) is tuple:
            temp_operator_formats += list(map(tuple, permutations(of)))

    return list(set(operator_format + temp_operator_formats))
    



def is_num(s, dtype):
    if dtype == "int" or dtype == "bool":
        try:
            int(s)
        except ValueError:
            return False
    elif dtype == "float":
        try:
            float(s)
        except ValueError:
            return False
    else:
        raise NotImplementedError(dtype)

    return True






def main(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("file_path",  help="Select file path", type=Path)
    args = parser.parse_args()
    main(args)
