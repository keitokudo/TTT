import argparse
from pathlib import Path
import pickle
import random
import json
import os

import _jsonnet as jsonnet
from logzero import logger

class NumberGenerator:
    def __init__(self, config_file_path:Path):

        config = json.loads(
            jsonnet.evaluate_file(
                str(config_file_path.expanduser()),
                ext_vars=dict(os.environ),
            )
        )
        
        self.seed = config["seed"]
        self.random_module = random.Random(self.seed)

        self.min_num = config["min_num"]
        self.max_num = config["max_num"]
        self.output_dir_path = Path(config["output_dir_path"]).expanduser()
        assert self.output_dir_path.is_dir(), f"Selected path \"{self.output_dir_path}\" is not directory..."
        
        assert len(config["contents"]) == len(config["split_rate"]), "Length dose not match..."
        self.split_rate = {k: v for k, v in zip(config["contents"], config["split_rate"])}
        self.splited_num_list = {k: None for k in config["contents"]}

        assert len(self.split_rate) == len(self.splited_num_list), "Splited contents length dose not match..."

        
    def __call__(self):
        num_list = list(range(self.min_num, self.max_num))
        self.random_module.shuffle(num_list)

        strat_idx = 0
        for k, v in self.split_rate.items():
            num_set_size = int(len(num_list) * (v  / sum(self.split_rate.values())))
            self.splited_num_list[k] = num_list[strat_idx:num_set_size + strat_idx]
            strat_idx = num_set_size + strat_idx

        _reminder_number = num_list[strat_idx:]

        #save
        for k, v in self.splited_num_list.items():
            with (self.output_dir_path / f"{k}_numbers.pkl").open(mode="wb") as f:
                pickle.dump(v, f)
                logger.info(f"{k}: {len(v)}")


def main(args):
    number_generator = NumberGenerator(args.config_file_path)
    number_generator()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file_path", help="Select file path", type=Path)
    args = parser.parse_args()
    main(args)
