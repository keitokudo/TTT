from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BasicOperator(ABC):
    arg_format = None
    

    @abstractmethod
    def __call__(self, arg_list: List[str], state: Dict[str, Any]):
        raise NotImplementedError()

    @abstractmethod
    def get_representation(self, arg_list: List[str], state: Dict[str, Any]):
        raise NotImplementedError()

            

        
