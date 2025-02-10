def is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def gather_numbers(text, return_as_str=False):
    splited_text = text.split()
    numbers = []
    for token in splited_text:
        if is_int(token):
            numbers.append(token)
    if return_as_str:
        return numbers
    else:
        return [int(n) for n in numbers]
    


class ConstraintBase:
    def __init__(self, *args, **kwargs):
        pass
    
    def __call__(self, passage, question, answer, scratchpad=None):
        raise NotImplementedError



class NoneMinusConstraint(ConstraintBase):
    def __init__(self, ignore_passage=False, ignore_answer=False, ignore_scratchpad=False):
        super().__init__()
        self.ignore_passage = ignore_passage
        self.ignore_answer = ignore_answer
        self.ignore_scratchpad = ignore_scratchpad
        
    def __call__(self, passage, question, answer, scratchpad=None):
        if not self.ignore_passage:
            if any(n < 0 for n in gather_numbers(passage, return_as_str=False)):
                return False

        if not self.ignore_answer:
            if any(int(n) < 0 for n in answer):
                return False

        if scratchpad is not None and not self.ignore_scratchpad:
            if any(n < 0 for n in gather_numbers(scratchpad, return_as_str=False)):
                return False

        return True


class DigitsConstraint(ConstraintBase):
    def __init__(self, n_passage_digits=None, n_answer_digits=None, n_scratchpad_digits=None):
        super().__init__()
        self.n_passage_digits = n_passage_digits
        self.n_answer_digits = n_answer_digits
        self.n_scratchpad_digits = n_scratchpad_digits
        
    def __call__(self, passage, question, answer, scratchpad=None):
        if any(len(n) != self.n_passage_digits for n in gather_numbers(passage, return_as_str=True)):
            return False

        for ans in answer:
            if len(str(ans)) != self.n_answer_digits:
                return False
            
        if scratchpad is not None:
            for sp in gather_numbers(scratchpad, return_as_str=True):
                if len(str(sp)) != self.n_scratchpad_digits:
                    return False

        return True


class NumberRangeConstraint(ConstraintBase):
    def __init__(
            self,
            passage_range=None,
            answer_range=None,
            scratchpad_range=None,
            only_last_answer=False,
    ):
        super().__init__()
        self.passage_range = passage_range
        self.answer_range = answer_range
        self.scratchpad_range = scratchpad_range
        self.only_last_answer = only_last_answer
        
    def __call__(self, passage, question, answer, scratchpad=None):
        if self.passage_range is not None:
            for n in gather_numbers(passage, return_as_str=False):
                if self.passage_range[0] <= n <= self.passage_range[1]:
                    continue
                else:
                    return False
                
        if self.answer_range is not None:
            if self.only_last_answer:
                answer = answer[-1:]
                
            for ans in map(int, answer):
                if self.answer_range[0] <= ans <= self.answer_range[1]:
                    continue
                else:
                    return False
                
        if scratchpad is not None and self.scratchpad_range is not None:
            for sp in gather_numbers(scratchpad, return_as_str=False):
                if self.scratchpad_range[0] <= sp <= self.scratchpad_range[1]:
                    continue
                else:
                    return False
        return True


class VariableNameConstraint(ConstraintBase):
    def __init__(self, variable_names=None, positions=None):
        super().__init__()
        self.variable_names = variable_names
        self.positions = positions
        assert len(self.variable_names) == len(self.positions)
        
    def __call__(self, passage, question, answer, scratchpad=None):
        for i in self.positions:
            if question[i][0] not in self.variable_names:
                return False
        return True
