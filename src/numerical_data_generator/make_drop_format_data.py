import json
import random
import itertools
import os
import argparse
from itertools import count
from more_itertools import chunked
from copy import deepcopy
from numerical_data_generation import NumericDataGenarator


def is_num(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True



def read_template(filename):
    with open(filename, mode="r") as f:
        json_dict = json.load(f)
    return json_dict



def passage_questuion_generater(config_filepath):
    pwd = os.path.dirname(__file__)
    
    template_dict = read_template(os.path.join(pwd, "template/drop_template_one_passage.json"))
    qa_template = read_template(os.path.join(pwd, "template/drop_template_question.json"))
    
    numeric_data_ganerator = NumericDataGenarator(config_filepath=config_filepath)

    counter = count()
    for passage, question, answer in numeric_data_ganerator():
        new_passage_question = deepcopy(template_dict)
        new_passage_question["passage"] = passage

        for q, a in zip(question, answer):
            new_qa = deepcopy(qa_template)
            new_qa["query_id"] = str(next(counter))
            new_qa["question"] = q

            # 数字であればnumber(生成して解答)に, そうでなければspan(抽出での解答)とする
            if is_num(a):
                new_qa["answer"]["number"] = a
            else:
                new_qa["answer"]["spans"].append(a)
                
            new_passage_question["qa_pairs"].append(new_qa)
            
        
        yield new_passage_question

        



def drop_dataset_generater(save_filename, config_filepath):

    with open(save_filename, mode="w") as f:
        f.write("{\n")
        counter = count()
        is_first = True
        for passage_questuion in passage_questuion_generater(config_filepath):
            if not is_first:
                f.write(",\n")
                
            
            data_name = f"nfl_{next(counter)}"
            new_data = json.dumps({data_name: passage_questuion}, indent=4)
            new_data = new_data[1:-2]
            
            f.write(new_data)
            is_first = False
            
        f.write("\n}\n")

            

def main(args):
    drop_dataset_generater(args.output_filepath, args.config_filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_filepath",  help="Select config file", type = str)
    parser.add_argument("output_filepath",  help="Select output file", type = str)
    args = parser.parse_args()
    main(args)
