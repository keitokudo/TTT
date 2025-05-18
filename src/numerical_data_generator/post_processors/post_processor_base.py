class PostProcessorBase:
    def __init__(self, **kwargs):
        pass
    
    def __call__(self, operator_config, assignment_configs):
        raise NotImplementedError


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
