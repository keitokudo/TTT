import random
import json

from .post_processor_base import PostProcessorBase

__all__ = ["IdenticalPostProcessor"]

class IdenticalPostProcessor(PostProcessorBase):
    def __call__(self, operator_config, assignment_configs):
        success = True
        return operator_config, assignment_configs, success


class EditSecondFormulaNumber(PostProcessorBase):
    def __init__(self, **kwargs):
        self.random_module = random.Random(kwargs["seed"])
        
    def insertion(self, rang1, rang2):
        return (
            max(rang1[0], rang2[0]),
            min(rang1[1], rang2[1])
        )

    def flip(self, rang):
        return (
            -1 * rang[1],
            -1 * rang[0]
        )
    
    def __call__(self, operator_config, assignment_configs):
        success = True
        
        formula2_boundary = (
            0 - int(assignment_configs[1]["format"][0]),
            9 - int(assignment_configs[1]["format"][0])
        )
        if assignment_configs[1]["type"] == "Sub":
            formula2_boundary = self.flip(formula2_boundary)
        formula2_boundary = self.insertion(formula2_boundary, (0, 9))

        
        change_number_boundary = (
            formula2_boundary[0] - int(assignment_configs[0]["format"][0]),
            formula2_boundary[1] - int(assignment_configs[0]["format"][0])
        )
        if assignment_configs[1]["type"] == "Sub":
            change_number_boundary = self.flip(change_number_boundary)
        change_number_boundary = self.insertion(change_number_boundary, (0, 9))
        
        # If the range is invalid or the only one option is available, fail
        if change_number_boundary[0] >= change_number_boundary[1]:
            success = False
            return operator_config, assignment_configs, success
        
        candidates = set(range(change_number_boundary[0], change_number_boundary[1] + 1))
        original_number = int(assignment_configs[0]["format"][1])
        try:
            candidates.remove(original_number)
        except KeyError:
            pass
        if len(candidates) == 0:
            success = False
            return operator_config, assignment_configs, success
        
        change_number = self.random_module.choice(list(candidates))
        assignment_configs[0]["format"][1] = str(change_number)
        return operator_config, assignment_configs, success

    
    
class EditSecondFormulaNumberWithDistractor(PostProcessorBase):
    def __init__(self, **kwargs):
        self.random_module = random.Random(kwargs["seed"])
        
    def insertion(self, rang1, rang2):
        return (
            max(rang1[0], rang2[0]),
            min(rang1[1], rang2[1])
        )

    def flip(self, rang):
        return (
            -1 * rang[1],
            -1 * rang[0]
        )
    
    def __call__(self, operator_config, assignment_configs):
        success = True
        
        formula2_boundary = (
            0 - int(assignment_configs[2]["format"][0]),
            9 - int(assignment_configs[2]["format"][0])
        )
        if assignment_configs[2]["type"] == "Sub":
            formula2_boundary = self.flip(formula2_boundary)
        formula2_boundary = self.insertion(formula2_boundary, (0, 9))

        
        change_number_boundary = (
            formula2_boundary[0] - int(assignment_configs[1]["format"][0]),
            formula2_boundary[1] - int(assignment_configs[1]["format"][0])
        )
        if assignment_configs[1]["type"] == "Sub":
            change_number_boundary = self.flip(change_number_boundary)
        change_number_boundary = self.insertion(change_number_boundary, (0, 9))
        
        # If the range is invalid or the only one option is available, fail
        if change_number_boundary[0] >= change_number_boundary[1]:
            success = False
            return operator_config, assignment_configs, success

        candidates = set(range(change_number_boundary[0], change_number_boundary[1] + 1))
        original_number = int(assignment_configs[1]["format"][1])
        try:
            candidates.remove(original_number)
        except KeyError:
            pass
        if len(candidates) == 0:
            success = False
            return operator_config, assignment_configs, success
        
        change_number = self.random_module.choice(list(candidates))
        assignment_configs[1]["format"][1] = str(change_number)
        return operator_config, assignment_configs, success
        

class EditFirstFormulaNumberWithDistractor(PostProcessorBase):
    def __init__(self, **kwargs):
        self.random_module = random.Random(kwargs["seed"])
        self.func_map = {
            "Add": lambda x, y: x + y,
            "Sub": lambda x, y: x - y,
        }
        
    def insertion(self, rang1, rang2):
        return (
            max(rang1[0], rang2[0]),
            min(rang1[1], rang2[1])
        )

    def flip(self, rang):
        return (
            -1 * rang[1],
            -1 * rang[0]
        )

    def compute(self, func_name, operand1, operand2):
        return self.func_map[func_name](operand1, operand2)
    
    def __call__(self, operator_config, assignment_configs):
        success = True
        
        second_formula_ans = self.compute(
            assignment_configs[2]["type"],
            int(assignment_configs[1]["format"][0]),
            int(assignment_configs[1]["format"][1])
        )

        if assignment_configs[2]["type"] == "Add":
            formula2_boundary = (
                0 - second_formula_ans,
                9 - second_formula_ans
            )
        if assignment_configs[2]["type"] == "Sub":
            formula2_boundary = (
                0 + second_formula_ans,
                9 + second_formula_ans
            )
            
        formula2_boundary = self.insertion(formula2_boundary, (0, 9))
        candidates = set(range(formula2_boundary[0], formula2_boundary[1] + 1))
        original_number = int(assignment_configs[2]["format"][0])
        try:
            candidates.remove(original_number)
        except KeyError:
            pass
        if len(candidates) == 0:
            success = False
            return operator_config, assignment_configs, success
        
        change_number = self.random_module.choice(list(candidates))
        assignment_configs[2]["format"][0] = str(change_number)
        return operator_config, assignment_configs, success
