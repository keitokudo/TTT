import os
import json
import itertools
import string
import argparse
from copy import deepcopy
from pathlib import Path
from itertools import permutations, chain, count
from pprint import pprint
from random import Random
import pickle


import _jsonnet as jsonnet
from logzero import logger

from . import operators
from .operators import *
from .numerical_question import NumericQuestion
from . import preprocess



def make_slice(slice_str:str):
    return slice(
        *map(
            int,
            slice_str.replace(" ", "").split(":")
        )
    )


class NumericDataGenarator:
    def __init__(self, config_filepath):
        self.config_dict = json.loads(
            jsonnet.evaluate_file(
                str(config_filepath),
                ext_vars=dict(os.environ),
            )
        )
        
        logger.info("Data Config:\n" + json.dumps(self.config_dict, indent=2))
        
        if not (self.config_dict["seed"] == "None"):
            assert type(self.config_dict["seed"]) is int, "Random seed is not int!"
            self.random_module = Random(self.config_dict["seed"])
            
        else:
            self.random_module = Random(42)
        
        assert os.environ.get('PYTHONHASHSEED') == "0", "Set enviroment variable \"PYTHONHASHSEED\" = \"0\""
        
        assert ("symbol_selection_slice" in self.config_dict) ^ ("number_of_symbols" in self.config_dict), \
            "Specify either \"symbol_selection_slice\" or \"number_of_symbols\""
        
        if "symbol_selection_slice" in self.config_dict:
            selection_slice = make_slice(
                self.config_dict["symbol_selection_slice"]
            )
            self.symbols = string.ascii_lowercase[selection_slice]
            
        elif "number_of_symbols" in self.config_dict:
            self.number_of_symbols = self.config_dict["number_of_symbols"]
            assert 2 <= self.number_of_symbols <= 26, \
                "number_of_symbols is out of range"        
            self.symbols = string.ascii_lowercase[:self.number_of_symbols]
        else:
            raise RuntimeError(
                "Specify either \"symbol_selection_slice\" or \"number_of_symbols\""
            )

        logger.info(
            f"Number of symbols: {len(self.symbols)}"
        )
        
        self.max_number_of_question = self.config_dict["max_number_of_question"]
        self.dtype = self.config_dict["dtype"]
        self.number_file_path = self.config_dict.get("number_file_path")
        
        if self.dtype == "bool":
            self.max_value = 1
            self.min_value = 0
        else:
            if self.number_file_path:
                assert self.config_dict.get("max_value") is None, "\"max_value\" is declared..."
                assert self.config_dict.get("min_value") is None, "\"min_value\" is declared..."
                self.max_value = None
                self.min_value = None
            else:
                self.max_value = self.config_dict["max_value"]
                self.min_value = self.config_dict["min_value"]
                
        self.generation_rules = self.config_dict["generation_rules"]
        self.operators_dict = {o: globals()[o] for o in operators.__all__} 
        self.output_type = self.config_dict["output_type"]

        if self.dtype=="int":
            if self.number_file_path:
                self.number_file_path = Path(self.number_file_path).expanduser()
                with self.number_file_path.open(mode="rb") as f:
                    number_list = pickle.load(f)
                assert len(number_list) > 0, "Load empty list..."
                assert all(type(n) is int for n in number_list) and type(number_list) is list, "Not integer list..."
                
                def rand_sample_from_list(a, b):
                    assert (a is None) and (b is None), "Input number must be None!"
                    return self.random_module.choice(number_list)
                
                self.random_func = rand_sample_from_list
            else:
                self.random_func = self.random_module.randint
                
        elif self.dtype=="float":
            self.random_func = lambda a, b : round(self.random_module.uniform(a, b), 2)
            
        elif self.dtype == "bool":
            def bool_rand_func(a, b):
                assert a == 0 and b == 1, "Not bool number..."
                return self.random_module.randint(0, 1)
            self.random_func = bool_rand_func
        else:
            raise NotImplementedError(
                f"Dtype \"{self.dtype}\" in config file is not defined"
            )


        
        self.shuffle_order = self.config_dict.get("shuffle_order", False)
        self.reverse_order = self.config_dict.get("reverse_order", False)
        assert not (self.shuffle_order and self.reverse_order), "shuffle_order and reverse_order can not be True at the same time."
        
        self.with_intermediate_reasoning_steps = self.config_dict.get(
            "with_intermediate_reasoning_steps", False
        )
        self.intermediate_reasoning_step_type = self.config_dict.get(
            "intermediate_reasoning_step_type", None
        )
        self.resoning_step_indices = self.config_dict.get("resoning_step_indices", None)
        
        self.preprocess = self.config_dict.get("preprocess")
        if self.preprocess is not None:
            self.preprocess_func = getattr(preprocess, self.preprocess)
        else:
            self.preprocess_func = getattr(preprocess, "no_preprocess")
        
    def format_assert(self, format_for_assertion, operator_acceptable_formats):
        """
        代入に用いる, 変数, 数値の形式が正しいか確認
        """
        
        if ("*num_var" in operator_acceptable_formats) or ("*var_num" in operator_acceptable_formats):
            return True
        
        all_num = all(map(lambda x: x == "num", format_for_assertion))
        all_var = all(map(lambda x: x == "var", format_for_assertion))
        
        
        for acceptable_format in operator_acceptable_formats:
            if type(acceptable_format) is tuple:
                
                if acceptable_format == format_for_assertion:
                    return True
                
            elif type(acceptable_format) is str:
                if acceptable_format.startswith("*num_var") or acceptable_format.startswith("*var_num"):
                    n = int(acceptable_format.split(":")[-1])
                    if len(format_for_assertion) >= n:
                        return True
                
                elif acceptable_format.startswith("*num") and all_num:
                    n = int(acceptable_format.split(":")[-1])
                    if len(format_for_assertion) >= n:
                        return True

                elif acceptable_format.startswith("*var") and all_var:
                    n = int(acceptable_format.split(":")[-1])
                    if len(format_for_assertion) >= n:
                        return True

                else:
                    tuple_acceptable_format = (acceptable_format, )
                    if tuple_acceptable_format == format_for_assertion:
                        return True
                    
            else:
                raise Exception(f"type : {type(acceptable_format)} can not be handled")
        

        return False

        
        
    def instantiate_format(self, possible_formats, assignment_format_type, temp_assignment_configs, commutative, generation_type, assumption_length=None):

        # (generation_type == "template") =>(ならば) (assumption_length is not  None)
        assert (not generation_type == "template") or (assumption_length is not  None), "Input \"assumption_length\" when you use \"generation\""
        
        if len(possible_formats) == 0:
            return []
        
        #リストの要素をタプル化
        possible_formats = list(map(lambda x: tuple(x) if type(x) is list else x, possible_formats))
        
        if commutative:
            #possible_formats += list(permutations(possible_formats))
            temp_possible_formats = []
            for pf in possible_formats:
                if not(type(pf) is tuple):
                    continue
                temp_possible_formats += list(map(tuple, permutations(pf)))

            # 同一のものを取り除く
            possible_formats = list(set(possible_formats + temp_possible_formats))
        
        possible_formats = sorted(possible_formats, key=hash)
        
        temp_assignment_config_variables = [tac["variable"] for tac in temp_assignment_configs]
        selected_format = self.random_module.choice(possible_formats)
        instantiated_format = []
        
        if type(selected_format) is tuple:

            # 代入に用いる, 変数, 数値の形式が正しいか確認
            format_for_assertion = tuple(map(lambda x : "var" if type(x) is int else x, selected_format))
            
            operator_acceptable_formats = self.operators_dict[assignment_format_type].arg_formats
            
            
            assert self.format_assert(format_for_assertion, operator_acceptable_formats), f"\"{assignment_format_type}\" is not support \"{format_for_assertion}\"."

            
            for elem in selected_format:
                if elem == "num":
                    instantiated_format.append(str(self.random_func(self.min_value, self.max_value)))
                elif elem == "var":
                    instantiated_format.append(self.random_module.choices(temp_assignment_config_variables)[0])
                elif type(elem) is int:
                    assert generation_type=="template", "generation_rules:type = \"random\" can not use index selection."
                    # - でリストを参照する際に, temp_assignment_config_variables(リスト)の所望の位置にアクセスできるようにする. (temp_assignment_config_variablesは長さがその時点まで処理したものしか格納されていないため)
                    if elem < 0:
                        elem = assumption_length + elem
                    
                    instantiated_format.append(temp_assignment_config_variables[elem])
                else:
                    raise NotImplementedError()
        
        
        else:

            # 代入に用いる, 変数, 数値の形式が正しいか確認
            format_for_assertion = "var" if type(selected_format) is int else selected_format

            
            operator_acceptable_formats = self.operators_dict[assignment_format_type].arg_formats
            
            assert self.format_assert((format_for_assertion, ), operator_acceptable_formats), f"\"{assignment_format_type}\" is not support \"{format_for_assertion}\"."
            
            if selected_format == "num":
                instantiated_format.append(str(self.random_func(self.min_value, self.max_value)))
            elif selected_format == "var":
                instantiated_format.append(self.random_module.choices(temp_assignment_config_variables)[0])
            elif type(selected_format) is int:
                assert generation_type=="template", "generation_rules:type = \"random\" can not use index selection."
                if selected_format < 0:
                    elem = assumption_length + selected_format
                
                instantiated_format.append(temp_assignment_config_variables[selected_format])
            else:
                raise NotImplementedError()

                
        return instantiated_format





    
    

    def generator_of_template_configs(self, generation_rule):
        assert generation_rule["type"] == "template", "generation_rule's type is not match."
        
        while True:
            assignment_configs = []
            shuffled_symbol_list = self.random_module.sample(self.symbols, len(self.symbols))
        
            for assignment_format in generation_rule["assignment_format"]:
                temp_symbol = shuffled_symbol_list.pop()
                # None の時もfalseになる
                commutative = bool(assignment_format.get("commutative"))

                if type(assignment_format["type"]) is list:
                    selected_assignment_type = self.random_module.choice(assignment_format["type"])
                else:
                    selected_assignment_type = assignment_format["type"]
                    
                assignment_configs.append(
                    {
                        "variable" : temp_symbol,
                        "type" : selected_assignment_type,
                        "format" : self.instantiate_format(
                            assignment_format["format"],
                            selected_assignment_type,
                            assignment_configs,
                            commutative,
                            "template",
                            assumption_length=len(generation_rule["assignment_format"])
                        )
                    }
                )

                

            commutative = bool(generation_rule["operator"].get("commutative"))

            if type(generation_rule["operator"]["type"]) is list:
                selected_ope = self.random_module.choices(
                    generation_rule["operator"]["type"],
                    weights=generation_rule["operator"]["selection_probabilities"]
                )[0]
            else:
                selected_ope = generation_rule["operator"]["type"]

            
            operator_config = {
                "ope"  : selected_ope,
                "format" : self.instantiate_format(
                    generation_rule["operator"]["format"],
                    selected_ope,
                    assignment_configs,
                    commutative,
                    "template",
                    assumption_length=len(generation_rule["assignment_format"])
                )
            }
            
            
            self.set_order(assignment_configs)
            operator_config, assignment_configs = self.preprocess_func(
                operator_config,
                assignment_configs,
                self,
            )
            yield operator_config, assignment_configs
            

            

    def get_possible_assignment_format(self, assignment_format_list, generation_step_capacity, number_of_available_variable):
        """
        現在, 使用可能な変数への値の割り当て方法のリストを作成する. 
        条件1 :　残り使用可能なステップ数以下である
        条件2 :　既に定義されている変数の数が, 引数で使用される変数の数よりも多い
        """
        possible_assignment_format = []
        
        for assignment_format in assignment_format_list:
            if assignment_format["step_weight"] > generation_step_capacity:
                continue
            elif len(list(filter(lambda x: x=="var", assignment_format["format"]))) > number_of_available_variable:
                continue
            else:
                possible_assignment_format.append(assignment_format)
        
        return possible_assignment_format
                
        

        


    def generator_of_random_configs(self, generation_rule):
        assert generation_rule["type"] == "random", "generation_rule's type is not match."
        assert all(map(lambda x: len(x["format"]) == 1, generation_rule["assignment_format"])), "assignment_format of random generation must have \"format\" that length is 1."

        
        min_generation_step = generation_rule["reasning_step"]["min"]
        max_generation_step = generation_rule["reasning_step"]["max"]
        assert min_generation_step, "\"reasning_step:min\" must be longer than 2"

        
        # 最初は必ず変数の数値代入でなければならないので, その設定は(生成確率0でも良いので)configに含める. 
        substitution_num_step = next(filter(lambda x: x["type"] == "Substitution" and x["format"] == ["num"], generation_rule["assignment_format"]))["step_weight"]


        # yieldのループ
        while True:
            shuffled_symbol_list = self.random_module.sample(self.symbols, len(self.symbols))


            possible_format_list = generation_rule["assignment_format"]
            possible_format_selection_probability = [pf["probability"] for pf in possible_format_list]
            generation_step_capacity = self.random_module.randint(min_generation_step, max_generation_step)

            # 最初は必ず変数の数値代入でなければならない
            assignment_configs = [
                {
                    "variable" : shuffled_symbol_list.pop(),
                    "type" : "Substitution",
                    "format" : [str(self.random_func(self.min_value, self.max_value))]
                }
            ]         
            generation_step_capacity -= substitution_num_step
            
            assert generation_step_capacity > 0, "\"reasning_step:min\" must be more than \"step_wight\" of \"Substitution (format = num)\""


            
            # ステップ数が(generation_step_capacity)が尽きるまで, 変数の割り当て方法を決定する
            while True:
                
                possible_assignment_format = self.get_possible_assignment_format(generation_rule["assignment_format"], generation_step_capacity, len(assignment_configs))

                
                # ステップ数を使い切り, これ以上割り当てを行えなくなったらループを抜ける 
                if len(possible_assignment_format) == 0:
                    assert generation_step_capacity == 0, "You must include rule that has \"step_weight = 1\""
                    break
                
                
                possible_assignment_weights = [paf["probability"] for paf in possible_assignment_format]
                assignment_format = self.random_module.choices(possible_assignment_format, weights=possible_assignment_weights)[0]

                temp_symbol = shuffled_symbol_list.pop()
                # None の時もfalseになる
                commutative = bool(assignment_format.get("commutative"))


                if type(assignment_format["type"]) is list:
                    selected_assignment_type = self.random_module.choice(assignment_format["type"])
                else:
                    selected_assignment_type = assignment_format["type"]
                
                
                assignment_configs.append(
                    {
                        "variable" : temp_symbol,
                        "type" : selected_assignment_type,
                        "format" : self.instantiate_format(assignment_format["format"], selected_assignment_type, assignment_configs, commutative, generation_type="random")
                    }
                )

                # 使用可能な残りのステップ数を減らす
                generation_step_capacity -= assignment_format["step_weight"]
                
                
                
            commutative = bool(generation_rule["operator"].get("commutative"))

            if type(generation_rule["operator"]["type"]) is list:
                selected_ope = self.random_module.choices(generation_rule["operator"]["type"], weights=generation_rule["operator"]["selection_probabilities"])[0]
            else:
                selected_ope = generation_rule["operator"]["type"]

            
            operator_config = {
                "ope"  : selected_ope,
                "format" : self.instantiate_format(generation_rule["operator"]["format"], selected_ope, assignment_configs, commutative, generation_type="random")
            }

            self.set_order(assignment_configs)
            operator_config, assignment_configs = self.preprocess_func(
                operator_config,
                assignment_configs,
                self,
            )
            yield operator_config, assignment_configs


    def rand_enumerate(self, iterable):
        random_indexes = self.random_module.sample(range(len(iterable)), len(iterable))
        yield from zip(random_indexes, iterable)
            

    def set_order(self, assignment_configs):
        for i, conf in enumerate(assignment_configs):
            conf["reasning_index"] = i

        if self.shuffle_order:
            for i, conf in self.rand_enumerate(assignment_configs):
                conf["repr_index"] = i
        elif self.reverse_order:
            for i, conf in enumerate(reversed(assignment_configs)):
                conf["repr_index"] = i
        else:
            for i, conf in enumerate(assignment_configs):
                conf["repr_index"] = i
            

    def get_pqa_triple_from_configs(self, operator_config, assignment_configs, separate=False):
        #assignment_configs = sorted(assignment_configs, key=lambda d: d["reasning_index"])
        
        neumeric_question = NumericQuestion(
            operator_config,
            assignment_configs,
            self.output_type,
            with_intermediate_reasoning_steps=self.with_intermediate_reasoning_steps,
            intermediate_reasoning_step_type=self.intermediate_reasoning_step_type,
            resoning_step_indices=self.resoning_step_indices,
        )
        result = neumeric_question()

        
        separated_passage = result[0].split(", ")
        assert len(separated_passage) == len(assignment_configs), \
            "Passage length dose not match..."

        list_shuffled_passage, _ = zip(
            *sorted(
                zip(separated_passage, assignment_configs),
                key=lambda x: x[1]["repr_index"]
            )
        )
        
        
        if self.with_intermediate_reasoning_steps and (self.shuffle_order or self.reverse_order):
            answer = result[2][0]
            shuffled_answer_list = []
            for one_step in answer.split(" ; "):
                splited_step = one_step.split(", ")
                passage_part = splited_step[:len(separated_passage)]
                question_part = splited_step[len(separated_passage):]

                shuffled_passage_part, _  = zip(
                    *sorted(
                        zip(passage_part, assignment_configs),
                        key=lambda x: x[1]["repr_index"]
                    )
                )
                shuffled_answer_list.append(
                    ", ".join(list(shuffled_passage_part) + question_part)
                )
                

                result[2][0] = " ; ".join(shuffled_answer_list)
                
        if self.with_intermediate_reasoning_steps:
            assert len(result) == 4, "Result dose not include intermediate reasoning steps..."
            result = (", ".join(list_shuffled_passage), result[1], result[2], result[3])
            if separate:
                result = [(result[0], q, a, result[3]) for q, a in zip(result[1], result[2])]
        else:
            result = (", ".join(list_shuffled_passage), result[1], result[2])
            if separate:
                result = [(result[0], q, a) for q, a in zip(result[1], result[2])]
        
        return result


    
            
    def __call__(self, generate_config=False):
        generator_list = []

        #各ルールに基づいたジェネレータの作成
        for generation_rule in self.generation_rules:
            if generation_rule["type"] == "random":
                generator_list.append(self.generator_of_random_configs(generation_rule))
            elif generation_rule["type"] == "template":
                generator_list.append(self.generator_of_template_configs(generation_rule))
            else:
                error_rule_name = generation_rule["type"]
                raise Exception(f"rule \"{error_rule_name}\" is not defined")
        
        selection_weigths = [generation_rule["selection_probability"] for generation_rule in self.generation_rules]


        if self.max_number_of_question == "inf":
            counter = count()
        else:
            counter = range(self.max_number_of_question)
        
        
        if not generate_config: 
            for i in counter:
                temp_generator = self.random_module.choices(generator_list, weights=selection_weigths)[0]
                yield self.get_pqa_triple_from_configs(*next(temp_generator))
        else:
            for i in counter:
                yield next(self.random_module.choices(generator_list, weights=selection_weigths)[0])
        
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_filepath",  help="Select config file", type = Path)
    args = parser.parse_args()
    
    
    N = NumericDataGenarator(config_filepath=args.config_filepath)
    g = N(generate_config=True)
    for data in g:
        operator_config, assignment_configs = data
        pprint(operator_config)
        pprint(assignment_configs)
        print(N.get_pqa_triple_from_configs(*data))
        
