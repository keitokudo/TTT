from pathlib import Path
import pickle
import shutil
import json
from copy import deepcopy

from more_itertools import ilen
from logzero import logger
from tqdm import trange

from numerical_data_generator import NumericDataGenarator
from numerical_data_generator import constraints
from numerical_data_generator.post_processors import post_processors


class DatasetGeneratorBase():
    def __init__(
            self,
            number_of_data:int,
            save_file_path:Path,
            exclude_dataset_paths=None,
    ):
        
        raise NotImplementedError()
        
    def prepare_data(self):        
        raise NotImplementedError()

    
    def in_exclude_datasets(self, pq_tuple):
        assert len(pq_tuple) == 2, "len(pq_tuple) != 2 ..."
        return any(pq_tuple in passage_question_set for passage_question_set in self.exclude_dataset_sets)

    def in_myself(self, pq_tuple):
        assert len(pq_tuple) == 2, "len(pq_tuple) != 2 ..."
        return pq_tuple in self.passage_question_set

    def isdisjoint(self, passage_question_set):
        return self.passage_question_set.isdisjoint(passage_question_set)


class DentakuDatasetGeneratorBase(DatasetGeneratorBase):
    def __init__(
            self,
            data_config_file_path,
            number_of_data:int,
            save_file_path:Path,
            exclude_dataset_paths=None,
            constraint_cls_names=None,
    ):
        
        raise NotImplementedError()

    def formula_configs_loader(self):
        assert self.fconf_file_path.exists(), f"\"{self.fconf_file_path}\" is not exist..."
        with self.fconf_file_path.open(mode="rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break
                
class FormulaConfigsLoader():
    def __init__(self, data_config_file_path):
        self.data_config_file_path = data_config_file_path
        self.fconf_file_path = Path(self.data_config_file_path["fconf_file_path"])
        self.config_dict = {}
        assert self.fconf_file_path.exists(), f"\"{self.fconf_file_path}\" is not exist..."
        
    def __call__(self, generate_config=True):
        with self.fconf_file_path.open(mode="rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break



    
class DentakuDatasetGenerator(DentakuDatasetGeneratorBase):
    def __init__(
            self,
            data_generator_name:str,
            data_config_file_path,
            number_of_data:int,
            save_file_path:Path,
            exclude_dataset_paths=None,
    ):
        self.save_file_path = save_file_path
        self.exclude_dataset_paths = [] if exclude_dataset_paths is None else exclude_dataset_paths
        self.exclude_dataset_sets = []
        
        for path in self.exclude_dataset_paths:
            with (path / "set.pkl").open(mode="rb") as f:
                self.exclude_dataset_sets.append(pickle.load(f))


        self.data_config_file_path = data_config_file_path
        data_generator_cls = globals().get(data_generator_name)
        self.question_generator = data_generator_cls(data_config_file_path)
        
        assert self.question_generator.max_number_of_question == "inf", \
            "Question_generator config's \"max_number_of_question\" must be \"inf\""
        
        self.number_of_data = number_of_data
        self.passage_question_set = None
        self.formula_configs = None

        self.constraints = []
        constraints_def = self.question_generator.config_dict.get("constraints", {})
        for name, kwargs in constraints_def.items():
            cls = getattr(constraints, name)
            self.constraints.append(cls(**kwargs))

        self.post_processors = []
        post_processor_config = self.question_generator.config_dict.get(
            "post_processors", {}
        )
        for name, kwargs in post_processor_config.items():
            cls = getattr(post_processors, name)
            self.post_processors.append(cls(**kwargs))

        
        
    def __call__(self):
        self.save_file_path.mkdir(exist_ok=True)
        shutil.copy(
            self.data_config_file_path,
            self.save_file_path / self.data_config_file_path.name
        )
        
        self.set_file_path = self.save_file_path / "set.pkl"
        self.fconf_file_path = self.save_file_path / "fconf.pkl"
        self.raw_data_file_path = self.save_file_path / "raw_dataset.jsonl"
        self.passage_question_set = set()
        
        edited_output_files = [
            (self.save_file_path / f"raw_edited_{post_processor.__class__.__name__}_{i}.jsonl").open(mode="w")
            for i, post_processor in enumerate(self.post_processors)
        ]
        with self.fconf_file_path.open(mode="wb") as f_fconf, \
             self.raw_data_file_path.open(mode="w") as f_raw:
            
            try:
                for i in trange(self.number_of_data):
                    while True:
                        base_operator_config, base_assignment_configs = next(
                            self.question_generator(generate_config=True)
                        )
                        
                        edited_config_list = []
                        success = True
                        for post_processor in self.post_processors:
                            operator_config, assignment_configs, success = post_processor(
                                deepcopy(base_operator_config),
                                deepcopy(base_assignment_configs)
                            )
                            if not success:
                                break
                            edited_config_list.append((operator_config, assignment_configs))
                        if not success:
                            continue

                        base_pqa_triple_list = self.question_generator.get_pqa_triple_from_configs(
                            base_operator_config,
                            base_assignment_configs,
                            separate=False
                        )

                        base_pq_tuple = (
                            base_pqa_triple_list[0],
                            " ".join(base_pqa_triple_list[1])
                        )
                        if any(not constraint(*base_pqa_triple_list) for constraint in self.constraints):
                            continue

                        if self.in_exclude_datasets(base_pq_tuple) or self.in_myself(base_pq_tuple):
                            continue


                        edited_pqa_triple_lists = []
                        edited_pq_tuples = []
                        for operator_config, assignment_configs in edited_config_list:
                            pqa_triple_list = self.question_generator.get_pqa_triple_from_configs(
                                operator_config,
                                assignment_configs,
                                separate=False
                            )
                            pq_tuple = (
                                pqa_triple_list[0],
                                " ".join(pqa_triple_list[1])
                            )
                            if any(not constraint(*pqa_triple_list) for constraint in self.constraints):
                                break

                            if self.in_exclude_datasets(pq_tuple) or self.in_myself(pq_tuple):
                                break

                            edited_pqa_triple_lists.append(pqa_triple_list)
                            edited_pq_tuples.append(pq_tuple)
                        else:
                            break


                    # Only passage and question
                    self.passage_question_set.add(base_pq_tuple)
                    pickle.dump((base_operator_config, base_assignment_configs), f_fconf)

                    base_jsonl_data = {
                        "passage": base_pqa_triple_list[0],
                        "question": base_pqa_triple_list[1],
                        "answer": base_pqa_triple_list[2],
                    }

                    # When include scratchpad
                    if len(base_pqa_triple_list) == 4:
                        base_jsonl_data["scratchpad"] = base_pqa_triple_list[3]

                    print(
                        json.dumps(
                            base_jsonl_data,
                            ensure_ascii=False
                        ),
                        file=f_raw
                    )
                    
                    for edited_pqa_triple_list, edited_pq_tuple, (operator_config, assignment_configs), f in zip(
                        edited_pqa_triple_lists,
                        edited_pq_tuples,
                        edited_config_list,
                        edited_output_files
                    ):
                        self.passage_question_set.add(edited_pq_tuple)
                        
                        edited_jsonl_data = {
                            "passage": edited_pqa_triple_list[0],
                            "question": edited_pqa_triple_list[1],
                            "answer": edited_pqa_triple_list[2],
                        }

                        # When include scratchpad
                        if len(edited_pqa_triple_list) == 4:
                            edited_jsonl_data["scratchpad"] = edited_pqa_triple_list[3]

                        print(
                            json.dumps(
                                edited_jsonl_data,
                                ensure_ascii=False
                            ),
                            file=f
                        )

                    
            finally:
                for f in edited_output_files:
                    f.close()
                    
        with self.set_file_path.open(mode="wb") as f:
            pickle.dump(self.passage_question_set, f)
        
        assert ilen(self.formula_configs_loader()) == self.number_of_data, \
            "formula_configs size dosen't match..."
        logger.info("Success to generate dataset!")
        
    


