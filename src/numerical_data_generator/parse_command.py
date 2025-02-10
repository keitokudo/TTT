from copy import deepcopy
import string
from typing import List, Dict
import re


class ParseCommandBase:
    def __call__(
            self,
            formula:str,
            stack:List[str],
            state:Dict[str, str],
            splited_passage: List[str],
            calc_result=None,
    ):
        raise NotImplementedError()
    

class Pop(ParseCommandBase):
    def __call__(
            self,
            formula:str,
            stack:List[str],
            state:Dict[str, str],
            splited_passage: List[str],
            calc_result=None,
    ):
        assert formula == "", "\"formula\" must be blank string..."
        variable = stack.pop()
        for passage in splited_passage:
            if passage.startswith(variable):
                next_formula = passage
                break
        else:
            raise ValueError(
                f"{variable} is not defined in {splited_passage}"
            )
        return next_formula, stack, state

class Push(ParseCommandBase):
    def __call__(
            self,
            formula:str,
            stack:List[str],
            state:Dict[str, str],
            splited_passage: List[str],
            calc_result=None,
    ):
        stack.append(formula[0])
        for s in reversed(formula.replace(" ", "")[2:]):
            if s in string.ascii_lowercase:
                stack.append(s)
        
        return "", stack, state
    
    
class Resolve(ParseCommandBase):
    def __init__(self):
        self.pattern = r"(.)=(-*)(\d+)"
        
    def __call__(
            self,
            formula:str,
            stack:List[str],
            state:Dict[str, str],
            splited_passage: List[str],
            calc_result=None,
    ):
        formula_without_space = formula.replace(" ", "")
        assert re.match(self.pattern, formula_without_space) is not None  
        variable = formula_without_space[0]
        value = formula_without_space[2:]
        assert not variable in state
        state[variable] = value
        return "", stack, state

class Ref(ParseCommandBase):
    def __call__(
            self,
            formula:str,
            stack:List[str],
            state:Dict[str, str],
            splited_passage: List[str],
            calc_result=None,
    ):
        
        variable = [s for s in formula[1:] if s in string.ascii_lowercase][0]
        next_formula = f"{formula[0]}{formula[1:].replace(variable, state[variable], 1)}"
        return next_formula, stack, state
        
        
class Calc(ParseCommandBase):
    def __call__(
            self,
            formula:str,
            stack:List[str],
            state:Dict[str, str],
            splited_passage: List[str],
            calc_result=None,
    ):
        assert all(not (s in string.ascii_letters) for s in formula.replace(" ", "")[2:]), \
            "Variables are remained..."
        variable = formula[0]
        ans = str(calc_result[variable])
        next_formula = f"{variable} = {ans}"
        return next_formula, stack, state




