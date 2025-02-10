import argparse
import os
from pathlib import Path
from datetime import datetime
import pickle
from multiprocessing import Pool, Process, Queue
from itertools import cycle
from collections import defaultdict

import torch
from transformers import AutoTokenizer
import json
from tqdm import tqdm, trange
from logzero import logger
import pandas as pd
import wandb
from lightning.pytorch import seed_everything

import torch_prover

def count_consecutive_last_eos_token_ids(decoded_id: torch.Tensor, eos_token_id: int):
    count = -1
    for i in range(len(decoded_id) - 1, -1, -1):
        if decoded_id[i] == eos_token_id:
            count += 1
        else:
            break
    if count == -1:
        return 0
    else:
        return count

def load_hidden_state(args):
    args, hidden_states_dir, layer_id, sample_id, data, eos_token_id = args
    # if not data["correct"]:
    #     return None
    hidden_states_file_path = hidden_states_dir / str(sample_id) / "states" / f"{layer_id}.pt"
    meta_data_file_path = hidden_states_dir / str(sample_id) / "meta_data.pt"
    assert hidden_states_file_path.exists(), f"{hidden_states_file_path} is not exist..."
    assert meta_data_file_path.exists(), f"{meta_data_file_path} is not exist..."
    hidden_states = torch.load(hidden_states_file_path)
    meta_data = torch.load(meta_data_file_path)
    assert meta_data["id"] == sample_id, f"Expected {sample_id}, but got {meta_data['id']}..."
    
    if args.include_decoding_phase_hidden_states:
        layer_hidden_states = hidden_states.to(torch.float32)
    else:
        layer_hidden_states = hidden_states[:meta_data["input_length"]].to(torch.float32)

    if eos_token_id is not None:
        num_eos = count_consecutive_last_eos_token_ids(
            meta_data["decoded_id"], eos_token_id
        )
        remove_size =  None if num_eos == 0 else -num_eos
    else:
        remove_size = None
        
    input_length = meta_data["input_length"]

    return  {
        "layer_hidden_states": layer_hidden_states[:remove_size],
        "input_length": input_length,
    }

def async_hidden_state_loading(
        args, hidden_states_dir, layer_id, results_items, eos_token_id
):

    if args.num_workers is None:
        num_workers = (os.cpu_count() - 1) // len(args.gpu_ids)
    elif args.num_workers < 0:
        num_workers = os.cpu_count() + args.num_workers + 1
    else:
        num_workers = args.num_workers
        
    with Pool(num_workers) as p:
        for result in tqdm(
                p.imap(
                    load_hidden_state,
                    (
                        (args, hidden_states_dir, layer_id, int(i), data, eos_token_id)
                        for i, data in results_items
                    )
                ),
                total=len(results_items)
        ):
            yield result




def is_left_padded(attention_mask, tokenizer):
    return attention_mask[0] == 0


def get_model_cls(args):
    model_kwargs = {"args": args}
    return getattr(torch_prover, args.probing_method), model_kwargs


def torch_corrcoef(x, y):
    """
    Calculate correlation coefficient
    """
    input_matrix = torch.stack([x, y], dim=0)
    return torch.corrcoef(input_matrix)[0, 1]

    

def get_evaluation_metric(args):
    if args.evaluation_metric == "mse":
        return torch.nn.functional.mse_loss
    elif args.evaluation_metric == "mae":
        return torch.nn.functional.l1_loss
    elif args.evaluation_metric == "accuracy":
        return lambda x, y:  (x == y).float().mean()
    elif args.evaluation_metric == "corr":
        # Calculate correlation coefficient
        return torch_corrcoef
    else:
        raise NotImplementedError(f"Evaluation metric: {args.evaluation_metric} is not implemented...")


def get_layer_indices(args, hidden_states_dir):
    first_sample_hidden_states_dir = hidden_states_dir / "0" / "states"
    layer_indices = [
        int(p.stem) for p in sorted(
            first_sample_hidden_states_dir.glob("*.pt"),
            key=lambda x: int(x.stem)
        )
    ]
    assert args.layer_indices is None or args.layer_step is None, "Specify either layer indices or layer step..."
    if args.layer_indices is not None:
        layer_indices = args.layer_indices
    elif args.layer_step is not None:
        layer_indices = layer_indices[::args.layer_step]
    return layer_indices


def make_hidden_states_container(args, hidden_states_dir, layer_idx, results_items, dataset, eos_token_id):
    container = {}
    if not args.async_loading:
        for i, data in tqdm(results_items, total=len(results_items)):
            i = int(i)
            instance = dataset[i]

            if not args.ignore_correctness:
                if args.only_incorrect:
                    if data["correct"]:
                        continue
                else:
                    if not data["correct"]:
                        continue
                
            loaded_data = load_hidden_state(
                (args, hidden_states_dir, layer_idx, i, data, eos_token_id)
            )
            assert loaded_data is not None, f"{i} is not exist..."
            layer_hidden_states = loaded_data["layer_hidden_states"]
            input_length = loaded_data["input_length"]

            container[i] = {
                "layer_hidden_states": layer_hidden_states,
                "intermediate_answers": instance["intermediate_answers"],
                "input_length": input_length,
            }
    else:
        container = {
            i: d
            for i, d in enumerate(
                    async_hidden_state_loading(
                        args, hidden_states_dir, layer_idx, results_items, eos_token_id
                    )
            )
        }
        for i, d in container.items():
            if d is not None:
                d["intermediate_answers"] = dataset[i]["intermediate_answers"]
                
        # Remove None
        container = {k: v for k, v in container.items() if v is not None}       
    return container


def parallel_process_wrapper(func, args_list, return_queue=None):
    for args in args_list:
        output = func(*args)
        if return_queue is not None:
            return_queue.put(output)
            

def train_layer_loop(args, hidden_states_dir, layer_idx, results, dataset, eos_token_id, gpu_id=None):
    model_cls, model_kwargs = get_model_cls(args)
    device = torch.device(f"cuda:{gpu_id}") if gpu_id is not None else None
    
    if args.subset_size is not None:
        results_items = sorted(
            results["result"].items(),
            key=lambda x: str(x[0])
        )[:args.subset_size]
    else:
        results_items = sorted(
            results["result"].items(),
            key=lambda x: str(x[0])
        )
        
    container = make_hidden_states_container(
        args, hidden_states_dir, layer_idx, results_items, dataset, eos_token_id
    )
    logger.info(f"Number of correct data: {len(container)}")
    
    min_index = min(int(k) for k, v in results["result"].items() if v["correct"])
    # Validity check
    # assert all(container[min_index]["layer_hidden_states"].size() == v["layer_hidden_states"].size() for v in container.values()), "All hidden states should have same size..."
    assert all(len(container[min_index]["intermediate_answers"]) == len(v["intermediate_answers"]) for v in container.values()), "All intermediate answers should be same length..."
    assert all(container[min_index]["input_length"] == v["input_length"] for v in container.values()), "All input length should be same..."

    if args.positions is not None:
        positions = args.positions
    else:
        positions = range(container[min_index]["layer_hidden_states"].size(0))
    
    number_of_intermediate_answers = len(container[min_index]["intermediate_answers"])

    print("Positions: ", positions)
    print("Number of positions: ", len(positions))
    for p in tqdm(positions, desc="Position"):
        flattened_hidden_states = []
        for i, data in tqdm(container.items(), total=len(container)):
            hidden_states_before = data["layer_hidden_states"][p, :].view(-1)

            if gpu_id is None:
                hidden_states_before = hidden_states_before.numpy()
            flattened_hidden_states.append(hidden_states_before)


        for j in range(number_of_intermediate_answers):
            intermediate_answers = [
                data["intermediate_answers"][j] for data in container.values()
            ]
            if device is not None:
                model_kwargs["device"] = device

            model = model_cls(**model_kwargs)
            if model.model_type != "classification":
                intermediate_answers = [float(answer) for answer in intermediate_answers]
                
            logs = model.fit(flattened_hidden_states, intermediate_answers)    
            logger.info(f"Layer: {layer_idx}, Position: {p}, Intermediate answer: {j}")


            model_data = {
                "model": model,
                "kwargs": model_kwargs,
            }
            if gpu_id is None:
                extension = "pkl"
            else:
                extension = "pt"
            model_path = args.model_save_path / f"layer_{layer_idx}_position_{p}_intermediate_answer_{j}.{extension}"
            if gpu_id is None:
                with args.model_save_path.open("wb") as f:
                    pickle.dump(model_data, f)
            else:
                torch.save(model_data, model_path)

            if args.save_train_log:
                assert logs is not None, "Logs should not be None..."
                log_path = args.model_save_path / f"log_layer_{layer_idx}_position_{p}_intermediate_answer_{j}.json"
                with log_path.open("w") as f:
                    json.dump(logs, f)



def train(args, device=None):
    model_cls, model_kwargs = get_model_cls(args)
    
    with args.probing_train_result_file_path.open() as f:
        results = json.load(f)
    hidden_states_dir = Path(results["hidden_states_dir"])
    dataset_file_path = Path(results["corpus_path"]) if args.probing_train_dataset_file_path is None else args.probing_train_dataset_file_path
    eos_token_id = results["eos_token_id"]
    
    # Load dataset
    dataset = []
    with dataset_file_path.open() as f:
        for line in f:
            dataset.append(json.loads(line))
            
    layer_indices = get_layer_indices(args, hidden_states_dir)
    logger.info(f"Layer indices: {layer_indices}")
    
    args.model_save_path.mkdir(exist_ok=True, parents=True)
    if len(args.gpu_ids) <= 1:
        logger.info("Sequential training...")
        gpu_id = args.gpu_ids[0] if args.gpu_ids is not None else None
        for l in tqdm(layer_indices, desc="Layer"):
            train_layer_loop(
                args, hidden_states_dir, l, results, dataset, eos_token_id, gpu_id=gpu_id
            )
    else:
        logger.info("Parallel training...")
        id2args = defaultdict(list)
        id_and_gpu_id = list(enumerate(args.gpu_ids))
        for layer_idx, (i, gpu_id) in zip(layer_indices, cycle(id_and_gpu_id)):
            id2args[i].append(
                (args, hidden_states_dir, layer_idx, results, dataset, eos_token_id, gpu_id)
            )

        parent_processes = []
        for _, args_list in id2args.items():
            parent_process = Process(
                target=parallel_process_wrapper,
                args=(train_layer_loop, args_list, None)
            )
            parent_process.start()
            parent_processes.append(parent_process)            

        for parent_process in parent_processes:
            parent_process.join()
            
    logger.info("Training is done...")


def test_layer_loop(args, hidden_states_dir, layer_idx, results, dataset, eos_token_id, model_load_path, sample_tokens, gpu_id=None):
    model_cls, model_kwargs = get_model_cls(args)
    device = torch.device(f"cuda:{gpu_id}") if gpu_id is not None else None
    evaluation_metric = get_evaluation_metric(args)
    
    results_items = results["result"].items()
    
    container = make_hidden_states_container(
        args, hidden_states_dir, layer_idx, results_items, dataset, eos_token_id
    )
    logger.info(f"Number of correct data: {len(container)}")

    try:
        min_index = min(int(k) for k, v in results["result"].items() if v["correct"])
    except ValueError:
        logger.warning("All data is incorrect. Selecting minimum index regardless of correctness instead.")
        min_index = min(int(k) for k in results["result"].keys())
        
    # Validity check
    # assert all(container[min_index]["layer_hidden_states"].size() == v["layer_hidden_states"].size() for v in container.values()), "All hidden states should have same size..."
    assert all(len(container[min_index]["intermediate_answers"]) == len(v["intermediate_answers"]) for v in container.values()), "All intermediate answers should be same length..."
    assert all(container[min_index]["input_length"] == v["input_length"] for v in container.values()), "All input length should be same..."
    
    if args.positions is not None:
        positions = args.positions
    else:
        positions = range(container[min_index]["layer_hidden_states"].size(0))
    number_of_intermediate_answers = len(container[min_index]["intermediate_answers"])
    
    columns = []
    for p in tqdm(positions, desc="Position"):
        flattened_hidden_states = []
        is_errors = []
        
        for i, data in tqdm(container.items(), total=len(container)):
            try:
                hidden_states_before = data["layer_hidden_states"][p, :].view(-1)
            except IndexError:
                hidden_states_before = torch.zeros_like(
                    container[min_index]["layer_hidden_states"][p, :].view(-1)
                )
                is_errors.append(True)
            else:
                is_errors.append(False)
                
                
            if gpu_id is None:
                hidden_states_before = hidden_states_before.numpy()
            flattened_hidden_states.append(hidden_states_before)


        for j in range(number_of_intermediate_answers):
            intermediate_answers = [
                data["intermediate_answers"][j] for data in container.values()
            ]
            intermediate_answers = [float(answer) for answer in intermediate_answers]

            if gpu_id is None:
                extension = "pkl"
            else:
                extension = "pt"

            model_path = model_load_path / f"layer_{layer_idx}_position_{p}_intermediate_answer_{j}.{extension}"
            if not model_path.exists():
                logger.warning(f"{model_path} is not exist...")
                continue
            
            if gpu_id is None:
                with model_path.open("rb") as f:
                    model_data = pickle.load(f)
            else:
                model_data = torch.load(model_path)

            model = model_data["model"]
            model.set_device(device)
            prediction = model.predict(flattened_hidden_states).tolist()
            assert len(is_errors) == len(prediction), "Length of is_errors and prediction should be same..."
            prediction = [
                float("nan") if is_error else pred
                for is_error, pred in zip(is_errors, prediction)
            ]
            
            if gpu_id is None:
                score = float(evaluation_metric(intermediate_answers, prediction))
            else:
                prediction = torch.tensor(prediction, dtype=torch.float32, device=device)
                intermediate_answers = torch.tensor(
                    intermediate_answers, dtype=torch.float32, device=device
                )
                score = evaluation_metric(intermediate_answers, prediction).item()
                
            logger.info(f"Layer: {layer_idx}, Position: {p}, Intermediate answer: {j}, Score: {score}")
            
            column = {
                "layer": layer_idx,
                "position": p,
                "intermediate_answer": j,
                "score": score,
                "token": sample_tokens[p],
                "predictions": prediction if type(prediction) is list else prediction.tolist(),
                "answer": intermediate_answers if type(intermediate_answers) is list else intermediate_answers.tolist(),
                "sample_ids": list(container.keys()),
            }
            columns.append(column)
            
    return columns
    

            
def test(args, model_load_path):
    with args.probing_test_result_file_path.open() as f:
        results = json.load(f)
    hidden_states_dir = Path(results["hidden_states_dir"])
    dataset_file_path = Path(results["corpus_path"]) if args.probing_test_dataset_file_path is None else args.probing_test_dataset_file_path
    eos_token_id = results["eos_token_id"]
    tokenizer = AutoTokenizer.from_pretrained(results["tokenizer_name_or_path"])
    
    # Load dataset
    dataset = []
    with dataset_file_path.open() as f:
        dataset = [json.loads(line) for line in f]


    layer_indices = get_layer_indices(args, hidden_states_dir)
    logger.info(f"Layer indices: {layer_indices}")
    
    first_sample_meta_data_file_path = hidden_states_dir / "0" / "meta_data.pt"
    assert first_sample_meta_data_file_path.exists(), f"{first_sample_meta_data_file_path} is not exist..."
    meta_data = torch.load(first_sample_meta_data_file_path)

    if "remove_prompt_part_hidden_states" in meta_data:
        eos_position = meta_data["eos_pos"]
        meta_data["decoded_id"] = meta_data["decoded_id"][eos_position + 1:]

    if args.include_decoding_phase_hidden_states:
        sample_tokens = [
            tokenizer.convert_ids_to_tokens([i])[0]
            for i in meta_data["decoded_id"]
        ]
    else:
        sample_tokens = [
            tokenizer.convert_ids_to_tokens([i])[0]
            for i in meta_data["decoded_id"][:meta_data["input_length"]]
        ]
        
        
    evaluation_results = []
    if len(args.gpu_ids) <= 1:
        for l in tqdm(layer_indices, desc="Layer"):
            gpu_id = args.gpu_ids[0] if args.gpu_ids is not None else None
            for column in test_layer_loop(
                args, hidden_states_dir, l, results, dataset, eos_token_id, model_load_path, sample_tokens, gpu_id=gpu_id
            ):
                wandb.log(column)
                evaluation_results.append(column)
    else:
        id2args = defaultdict(list)
        id_and_gpu_id = list(enumerate(args.gpu_ids))
        for layer_idx, (i, gpu_id) in zip(layer_indices, cycle(id_and_gpu_id)):
            id2args[i].append(
                (args, hidden_states_dir, layer_idx, results, dataset, eos_token_id, model_load_path, sample_tokens, gpu_id)
            )
            
        parent_processes = []
        return_queue = Queue()
        for _, args_list in id2args.items():
            parent_process = Process(
                target=parallel_process_wrapper,
                args=(test_layer_loop, args_list, return_queue)
            )
            parent_process.start()
            parent_processes.append(parent_process)
            
        while True:
            try:
                output = return_queue.get(timeout=1)
            except:
                if all(not p.is_alive() for p in parent_processes):
                    break
                continue
            for column in output:
                wandb.log(column)
                evaluation_results.append(column)

            
        for parent_process in parent_processes:
            parent_process.join()

    df = pd.DataFrame(evaluation_results, columns=list(evaluation_results[0].keys()))
    args.output_file_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_json(args.output_file_path, orient='records', force_ascii=False, lines=True)
    
def main(args):
    if args.async_loading:
        raise NotImplementedError("Async loading has bug...")
    
    seed_everything(args.seed)

    if args.probing_train_result_file_path is not None:
        assert args.probing_train_result_file_path.exists(), f"{args.probing_train_result_file_path} is not exist..."
        assert args.model_save_path is not None, "Specify model save path..."
        
    if args.probing_test_result_file_path is not None:
        assert args.probing_test_result_file_path.exists(), f"{args.probing_test_result_file_path} is not exist..."
        assert args.output_file_path is not None, "Specify output file path..."

    
    # Setup wandb
    wandb.init(
        project=args.project,
        name=args.version,
        config=args,
    )   
    if args.model_load_path is not None:
        model_load_path = args.model_load_path
    else:
        logger.info("Start training...")
        train(args)
        model_load_path = args.model_save_path
    assert model_load_path.exists(), f"{model_load_path} is not exist..."
    
    if args.probing_test_result_file_path is not None:
        logger.info("Start testing")
        test(args, model_load_path)
    wandb.finish()
    
if __name__ == "__main__":
    initial_parser = argparse.ArgumentParser()
    initial_parser.add_argument("--probing_method", "-m", help="Select probing method", type=str, required=True)
    initial_args, unrecognized = initial_parser.parse_known_args()
    model_cls = getattr(torch_prover, initial_args.probing_method, None)
    
    parser = argparse.ArgumentParser()
    if model_cls is None:
        raise ValueError(f"{initial_args.probing_method} is not exist...")
    else:
        model_cls.add_args(parser)
        
    parser.add_argument("--probing_train_result_file_path", help="Specify file path", type=Path)
    parser.add_argument("--probing_test_result_file_path", help="Specify file path", type=Path)
    parser.add_argument("--probing_train_dataset_file_path", help="Specify file path", type=Path)
    parser.add_argument("--probing_test_dataset_file_path", help="Specify file path", type=Path)
    parser.add_argument("--output_file_path", "-o", help="Specify file path", type=Path)
    parser.add_argument("--probing_method", "-m", help="Specify method", type=str, required=True)
    parser.add_argument("--evaluation_metric", "-e", help="Specify evaluation metrics", type=str, choices=["mse", "mae", "corr", "accuracy"], default="accuracy")
    parser.add_argument("--gpu_ids", "-g", help="Specify GPU ID", type=int, nargs="+")
    parser.add_argument("--include_decoding_phase_hidden_states", "-i", action="store_true")
    parser.add_argument("--model_save_path", help="Specify file path", type=Path)
    parser.add_argument("--model_load_path", help="Specify file path", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", help="Specify project name", type=str, default="llm_numerical_reasoning_interpretation")
    parser.add_argument(
        "--version",
        "-v",
        help="Specify version",
        type=str,
        default="evaluation_result_{}".format(datetime.today().strftime("%Y%m%d%H%M%S"))
    )
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--async_loading", action="store_true")
    parser.add_argument("--layer_step", type=int)
    parser.add_argument("--layer_indices", type=int, nargs="+")
    parser.add_argument("--positions", type=int, nargs="+")
    parser.add_argument("--subset_size", type=int)
    parser.add_argument("--save_train_log", action="store_true")
    parser.add_argument("--only_incorrect", action="store_true")
    parser.add_argument("--ignore_correctness", action="store_true")
    args = parser.parse_args()
    main(args)
