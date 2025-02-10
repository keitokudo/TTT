from copy import deepcopy
from itertools import chain, tee, islice, takewhile
from collections import deque
import inspect
import string
from pprint import pprint
from dataclasses import dataclass
from collections import OrderedDict
from typing import List, Dict
import re

from . import operators
from .operators import *
from .basic_operator import BasicOperator
from . import parse_command

# Copyed from toolz
def sliding_window(seq, n):
    return zip(*(deque(islice(it, i), 0) or it for i, it in enumerate(tee(seq, n))))


@dataclass
class ParseStep:
    command: str
    formula: str
    stack: List[str]
    state: Dict[str, str]
    


class NumericQuestion:
    def __init__(
            self,
            operator_config,
            assignment_config,
            output_type,
            with_intermediate_reasoning_steps=False,
            intermediate_reasoning_step_type=None,
            resoning_step_indices=None,
    ):
        
        self.ope = operator_config["ope"]
        assert self.ope in operators.__all__, "operator \"{ope}\" is not defined"
        
        self.ope_args = operator_config["format"]

        self.operators_dict = {o: globals()[o] for o in operators.__all__}
        self.prog_env = {}
        self.prog_env.update(**self.operators_dict)

        self.assumptions = assignment_config
        self.prog_states = []
        self.substitution_operator_dict = {}
        self.final_state = None
        self.ans = None
        self.output_type = output_type
        self.calculated = False

        # 途中の推論過程の出力
        self.with_intermediate_reasoning_steps = with_intermediate_reasoning_steps
        self.intermediate_reasoning_step_type = intermediate_reasoning_step_type
        
        assert self.with_intermediate_reasoning_steps or self.intermediate_reasoning_step_type is None, \
            "If you don't use intermediate_reasoning_steps, Specify \"intermediate_reasoning_step_type\" None"
        self.resoning_step_indices = resoning_step_indices
        
        self.intermediate_states = None
        self.prog_history = None
        
        # assert (not self.with_intermediate_reasoning_steps) or self.output_type == "ask_last_question", "\"with_intermediate_reasoning_steps\" cannot use except output_type == \"ask_last_question\""
        

    def exec_program(self):

        self.intermediate_states = []
        self.prog_history = []
                
        for i, assumption in enumerate(self.assumptions):
            
            # 引数となっている変数の値を保持
            args_state = {s : self.prog_env.get(s) for s in assumption["format"] if self.prog_env.get(s) is not None}

            
            self.prog_states.append(args_state)
            
            variable = assumption["variable"]
            
            self.prog_env["state"] = args_state
            self.prog_env["arg_list"] = assumption["format"]
            
            prog = f"""ope{i} = {assumption["type"]}()\n{variable} = ope{i}(arg_list, state)\n{variable} = round({variable}, 2) if type({variable}) is float else {variable}"""

            self.intermediate_states.append(deepcopy(self.prog_env))
            self.prog_history.append(
                ";".join(prog.split("\n")[:-1])
            )
                        
            exec(prog, {}, self.prog_env)
            
            self.substitution_operator_dict[f"ope{i}"] = self.prog_env[f"ope{i}"]

        
        args_state = {
            s: self.prog_env.get(s)
            for s in self.ope_args if not (self.prog_env.get(s) is None)
        }
        
        self.prog_states.append(args_state)
        self.prog_env["state"] = args_state
        self.prog_env["arg_list"] = self.ope_args
        
        prog = f"ope_last = {self.ope}()\nans = ope_last(arg_list, state)\nans = round(ans, 2) if type(ans) is float else ans"
        
        self.intermediate_states.append(deepcopy(self.prog_env))
        self.prog_history.append(";".join(prog.split("\n")[:-1]))
        
        exec(prog, {}, self.prog_env)
        
        self.substitution_operator_dict["ope_last"] = self.prog_env["ope_last"]
        self.intermediate_states.append(deepcopy(self.prog_env))
        

    def _basic_operator_filter(self, obj):
        if inspect.isclass(obj):
            return issubclass(obj, BasicOperator)
        else:
            return issubclass(obj.__class__, BasicOperator)
        

    def assign_value(self):

        #プログラムの実行
        self.exec_program()
        
        self.ans = self.prog_env["ans"]
        self.final_state = deepcopy(self.prog_env)
        
        # プログラムの状態から余分なクラス, 変数定義の削除
        for s in chain(self.operators_dict.keys(), self.substitution_operator_dict.keys()):
            self.final_state.pop(s)
        
        self.final_state.pop("state")
        self.final_state.pop("arg_list")
        
        self.calculated = True
        
        

    def make_passage(self):
        passage_list = []
        for i, (assumption, state) in enumerate(
                zip(
                    self.assumptions,
                    self.prog_states[:-1]
                )
        ):
            
            one_substitution_passage = "{} = {}".format(
                assumption["variable"],
                self.substitution_operator_dict[f"ope{i}"].get_representation(
                    assumption["format"],
                    state
                )
            )
            
            passage_list.append(one_substitution_passage)

        return ", ".join(passage_list)


    def make_intermediate_step(self, passage:str, question:str):
        if self.intermediate_reasoning_step_type is None or \
           self.intermediate_reasoning_step_type == "basic":
            return self.make_basic_intermediate_step(passage, question)

        else:
            return self.make_parse_based_intermediate_step(
                passage,
                question,
                self.intermediate_reasoning_step_type,
            )


    def extract_intermediate_step_from_indices(self, reasoning_step_strs):
        if self.resoning_step_indices is None:
            return reasoning_step_strs
        else:
            return [reasoning_step_strs[i] for i in self.resoning_step_indices]
        
    def make_parse_based_intermediate_step(
            self,
            passage:str,
            question:str,
            reasoning_step_type:str,
    ):

        assert len(self.intermediate_states) == len(self.prog_history) + 1
        assert len(question.replace(" ", "")) == 2 and question[-1] == "=", \
            "This intermediate step type only support Check question..."
        cleaned_intermediate_states = [
            self._clean_state(state) for state in self.intermediate_states
        ]
        last_state = self._clean_state(self.intermediate_states[-1])
        
        ascii_lowercase_set = set(string.ascii_lowercase)
        variable_list = [
            assumption["variable"] for assumption in self.assumptions
        ]
        splited_passage = passage.split(", ")
        intermediate_steps = []

        # Set initial step
        command, formula, stack, state = "", "", [question[0]], OrderedDict()
        history = [
            ParseStep(
                command="Push",
                formula=formula,
                stack=stack,
                state=state,
            )
        ]
        
        #print("\n", passage, question)
        while True:
            command_str = self.select_command(formula, stack, state)
            """
            print(f"{command_str}: ", history[-1])
            import time
            try:
                time.sleep(0.1)
            except:
                import sys
                sys.exit()
            """
            cmd = getattr(parse_command, command_str)()
            formula, stack, state = cmd(
                formula,
                stack,
                state,
                splited_passage,
                last_state,
            )
            history.append(
                ParseStep(
                    command=cmd.__class__.__name__,
                    formula=formula,
                    stack=deepcopy(stack),
                    state=deepcopy(state),
                )
            )
            
            # 終了判定
            if formula == "" and len(stack) == 0:
                break

            
        if reasoning_step_type == "strict":
            reasoning_step_strs = []
            for hist in history:
                stack_str = ", ".join(hist.stack)
                state_str = ", ".join("{k} : {v}" for k, v in state.items())
                reasoning_step_strs.append(
                    f"{hist.formula} [ {stack_str} ]" + "{ " + state_str + " }"
                )

            reasoning_step_strs = self.extract_intermediate_step_from_indices(
                reasoning_step_strs
            )    
            return f"{' , '.join(reasoning_step_strs)} , {history[-1].state[question[0]]}"

        elif reasoning_step_type == "simple":
            raise NotImplementedError()
        
        elif reasoning_step_type == "simple_backtracking":
            reasoning_step_strs = [
                p.formula for p in history[1:] if p.formula != ""
            ]
            return " , ".join(
                self.extract_intermediate_step_from_indices(reasoning_step_strs)
            )
        
        elif reasoning_step_type == "more_simple":
            reasoning_step_strs = [] 
            history_iter = iter(history[1:])
            paragraphs = []

            # Pop~PushとPop~resolveに分割する(段落に分ける)
            while True:
                paragraph = []
                for p in history_iter:
                    paragraph.append(p)
                    if p.command == "Push" or p.command == "Resolve":
                        break
                    
                    
                if len(paragraph) == 0:
                    break
                paragraphs.append(paragraph)
            
            
            for para in paragraphs:
                assert para[0].command == "Pop"
                if len(para) == 2 and (para[1].command == "Push" or para[1].command == "Resolve"):
                    continue

                for parse_step in para[:-1]:
                    reasoning_step_strs.append(parse_step.formula)

            # 代入のみの演算で推論が存在しない場合
            if len(reasoning_step_strs) == 0:
                reasoning_step_strs.append(paragraphs[0][0].formula)
                
            return " , ".join(
                self.extract_intermediate_step_from_indices(reasoning_step_strs)
            )
        
        elif reasoning_step_type == "only_substitution":
            reasoning_step_strs = [
                hist.formula for hist in history[1:] if hist.command == "Calc"
            ]
            return " , ".join(
                self.extract_intermediate_step_from_indices(reasoning_step_strs)
            )

        
    def select_command(
            self,
            formula:str,
            stack:List[str],
            state:Dict[str, str]
    ):
        ascii_lowercase_set = set(string.ascii_lowercase)
        
        if formula == "":
            return "Pop"
        
        
        variable = formula[0]
        formula_variable_args = [
            s for s in formula[2:] if s in ascii_lowercase_set
        ]

        if len(formula_variable_args) == 0:
            #calc or resolve
            # 演算子の有無で判定
            # a=1などの形式を判定
            if re.match(r"^[a-zA-Z]=(-*)(\d+)$", formula.replace(" ", "")):
                return "Resolve"
            else:
                return "Calc"
        
        
        if all(v in state for v in formula_variable_args):
            return "Ref"
        else:
            return "Push"
        
        raise RuntimeError(
            f"Invalid step state, {formula}, {stack}, {state}"
        )
        

    def make_basic_intermediate_step(self, passage:str, question:str):
        assert len(self.intermediate_states) == len(self.prog_history) + 1
        cleaned_intermediate_states = [
            self._clean_state(state) for state in self.intermediate_states
        ]
        ascii_lowercase_set = set(string.ascii_lowercase)
        variable_list = [
            assumption["variable"] for assumption in self.assumptions
        ]
        required_variable = [s for s in self.ope_args if s in ascii_lowercase_set]
        last_state = self.intermediate_states[-1]
        
        for s in deepcopy(required_variable):
            required_variable += self._get_required_variable(s)
        required_variable = list(set(required_variable))

        splited_passage = passage.split(", ")
        intermediate_step_passages = []
        first_process = True
        assert len(splited_passage) == len(self.assumptions) == len(list(sliding_window(cleaned_intermediate_states[:-1], 2))), \
            f"{len(splited_passage)} == {len(self.assumptions)} == {len(list(sliding_window(cleaned_intermediate_states, 2)))}"
        
        for i, assumption, formula, (state_before, state_after) in reversed(
                list(
                    zip(
                        range(len(self.assumptions)),
                        self.assumptions,
                        splited_passage,
                        sliding_window(cleaned_intermediate_states[:-1], 2)
                    )
                )
        ):
            
            if assumption["variable"] in required_variable:
                # 演算実行後の状態を生成
                after_passage = deepcopy(splited_passage)
                for j, asump in enumerate(self.assumptions):
                    variable = asump["variable"]
                    if variable in required_variable:
                        after_passage[j] = f"{variable} = {state_after[variable]}"

                # 最終状態を追加
                if first_process:
                    first_process = False
                    intermediate_step_passages.append(
                        ", ".join(after_passage)
                    )
                        
                # 現在見ている変数を削除
                required_variable.remove(assumption["variable"])
                temp_passage = deepcopy(after_passage)
                # 数字の代入のみの演算の時
                if temp_passage[i] == formula:
                    continue
                temp_passage[i] = formula
                temp_intermediate_step_passages = [", ".join(temp_passage)]
                
                for v in assumption["format"]:
                    # vが変数の時
                    if v in ascii_lowercase_set:
                        formula = formula.replace(v, str(state_before[v]), 1)
                        temp_passage[i] = formula
                        temp_intermediate_step_passages.append(
                            ", ".join(temp_passage)
                        )

                intermediate_step_passages += reversed(temp_intermediate_step_passages)
        
        intermediate_step_passages.reverse()
        
        # 質問文の簡約
        state_after = cleaned_intermediate_states[-1]
        formula = question[:-2]
        intermediate_step_questions = [formula]
        for v in self.ope_args:
            if v in ascii_lowercase_set:
                formula = formula.replace(v, str(state_after[v]), 1)
                intermediate_step_questions.append(formula)

        # 数字の代入のみの演算の時
        if intermediate_step_questions[-1] != str(state_after["ans"]):
            intermediate_step_questions.append(str(state_after["ans"]))


        result = []
        for p in intermediate_step_passages:
            result.append(f"{p}, {intermediate_step_questions[0]}")

        for q in intermediate_step_questions[1:]:
            result.append(f"{intermediate_step_passages[-1]}, {q}")

        return " , ".join(self.extract_intermediate_step_from_indices(result))
        
            

    def _get_required_variable(self, variable:str):
        result = []
        ascii_lowercase_set = set(string.ascii_lowercase)
        
        for assumption in reversed(self.assumptions):
            if (assumption["variable"] == variable) or (assumption["variable"] in result):
                for v in assumption["format"]:
                    if v in ascii_lowercase_set:
                        result.append(v)
        
        return list(set(result))
        
        
    def _clean_state(self, state):
        return {
            k: v
            for k, v in state.items()
            if not self._basic_operator_filter(v)
        }


    
    # 生成されたpassage, question, answerの3つ組を返す
    def make_pqa_triple(self):
        assert self.calculated, "Call assign_value meethod before this method."

        
        if self.output_type == "ask_last_question":
            
            passage = self.make_passage()
            assert not (self.ans is None), "Don't use No answer operation in \"ask_last_question\""
            answer = [str(self.ans)]
            question = [
                self.substitution_operator_dict["ope_last"].get_representation(
                    self.ope_args, self.prog_states[-1]
                ) + " ="
            ]

            if self.with_intermediate_reasoning_steps:
                intermediate_step_passage = self.make_intermediate_step(
                    passage,
                    question[0]
                )
                assert intermediate_step_passage[-len(answer[0]):] == answer[0], \
                    f"{intermediate_step_passage} but {answer[0]}"
                answer[0] = intermediate_step_passage
            else:
                intermediate_step_passage = None
                
            
        elif self.output_type == "ask_all_variables":
    
            passage = self.make_passage()
            self.final_state.pop("ans")
            question = [f"{k} =" for k in self.final_state.keys()]
            answer = list(map(str, self.final_state.values()))                

            if not (self.ans is None):
                last_question = self.substitution_operator_dict["ope_last"].get_representation(self.ope_args, self.prog_states[-1]) + " ="
                question.append(last_question)
                answer.append(str(self.ans))
                
            if self.with_intermediate_reasoning_steps:
                intermediate_step_passage = self.make_intermediate_step(
                    passage,
                    question[-1]
                )
                assert intermediate_step_passage[-len(answer[-1]):] == answer[-1], \
                    f"{intermediate_step_passage} but {answer[0]}"
            else:
                intermediate_step_passage = None

        else:
            raise NotImplementedError(f"output_type \"{self.output_type}\" is not defined!")
        
        return passage, question, answer, intermediate_step_passage



    def __call__(self):
        self.assign_value()
        return self.make_pqa_triple()
    
