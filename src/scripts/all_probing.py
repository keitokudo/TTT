import argparse
from pathlib import Path
import re
from typing import Tuple, List, Dict, Union, Optional
from collections import defaultdict

# ex. result_Yi-1.5-34B_seed_42_test_2step_in_order_20241024090136.json
def extract_tag(file_path: Path) -> str:
    file_name = file_path.name
    match_obj = re.match(r"result_(.*)_seed_(\d+)_(.*)_(\d+).json", file_name)
    if match_obj is None:
        raise ValueError(f"Invalid file name: {file_path}")

    tag = match_obj.group(3)

    if tag.startswith("test"):
        split = "test"
    elif tag.startswith("probing_train"):
        split = "probing_train"
    else:
        raise ValueError(f"Invalid tag: {tag}")

    tag_without_split = tag[len(split) + 1:]
    return split, tag_without_split
        
        

def main(args):
    # Gather all result file paths
    result_file_paths = list(args.log_dir.glob("result_*.json"))
    # Remove result_latest.json
    result_file_paths = [
        path.resolve() for path in result_file_paths if path.name != "result_latest.json"
    ]

    file_container = defaultdict(lambda: dict(test=None, probing_train=None))
    for path in result_file_paths:
        split, tag = extract_tag(path)
        file_container[tag][split] = path

    # Probing for not intervened results
    for tag, file_dict in file_container.items():
        if "edit" in tag:
            continue

        if file_dict["test"] is None or file_dict["probing_train"] is None:
            print(f"Skipping {tag}")
            continue

        command = f"zsh ./scripts/probing.sh {file_dict['probing_train']} {file_dict['test']}"
        print(command)

    # Probing for intervened results
    for tag, file_dict in file_container.items():
        if "edit" not in tag:
            continue
        
        if file_dict["test"] is None:
            print(f"Skipping {tag}")
            continue

        command = f"zsh ./scripts/probing_test.sh {file_dict['probing_train']} {file_dict['test']}"
        print(command)
        
        
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", "-l", help="Specify log directory", type=Path, required=True)
    #parser.add_argument("file_path", help="Specify file path", type=Path)
    #parser.add_argument("--use_gpu", action='store_true')
    args = parser.parse_args()
    main(args)
