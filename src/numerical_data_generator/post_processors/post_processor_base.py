class PostProcessorBase:
    def __init__(self, **kwargs):
        pass
    
    def __call__(self, operator_config, assignment_configs):
        raise NotImplementedError


