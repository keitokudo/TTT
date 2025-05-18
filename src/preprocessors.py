from pathlib import Path
import os
import mmap
from multiprocessing import Process, JoinableQueue, Manager
from itertools import islice

import numpy as np
from logzero import logger
from tqdm.auto import tqdm
from lightning.pytorch import seed_everything
from transformers import AutoTokenizer
from more_itertools import divide, ilen, chunked
import ujson
import cysimdjson

from utils import number_of_lines

from mmap_datset.indexed_dataset import MMapIndexedDatasetBuilder
from mmap_datset.file_chunker_utils import ChunkLineIterator, find_offsets
from dentaku_dataset_generator import DentakuDatasetGenerator


class DentakuPreProcessor:
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--data_generator",
            help="Specify data_generator",
            type=str,
            choices=["NumericDataGenarator", "FormulaConfigsLoader"],
            default="NumericDataGenarator",
        )
        parser.add_argument(
            "--config_file_path",
            help="Specify config file path",
            type=Path,
        )
        parser.add_argument(
            "--output_dir",
            help="Specify output_dir",
            type=Path,
            required=True
        )
        parser.add_argument(
            "--number_of_data",
            help="Specify number_of_data",
            type=int,
            required=True,
        )
        parser.add_argument(
            "--exclude_dataset_paths",
            help="Specify exclude_dataset_paths",
            type=Path,
            nargs="*",
        )
        parser.add_argument(
            "--seed",
            help="Specify seed",
            type=int,
            required=True,
        )
        parser.add_argument(
            "--eos_token",
            help="Specify eos_token",
            type=str,
            required=True,
        )
        parser.add_argument(
            "--pre_tokenization_method",
            help="Specify pre_tokenization_method",
            type=str,
            choices=["char", "word"],
            default="char",
        )
        parser.add_argument(
            "--few_shot_sample_dataset_path",
            help="Specify few_shot_sample_dataset_path",
            type=Path,
        )   
        parser.add_argument(
            "--number_of_few_shot_samples",
            help="Specify number_of_few_shot_samples",
            type=int,
        )
        parser.add_argument(
            "--question_first",
            help="Specify question_first",
            action="store_true",
        )
        parser.add_argument(
            "--prompt_pattern",
            help="Specify prompt_pattern",
            type=str,
            choices=["QA"],
        )

        
    def __init__(self, args):
        self.args = args
        seed_everything(self.args.seed)
        self.dataset_generator = DentakuDatasetGenerator(
            data_generator_name=self.args.data_generator,
            data_config_file_path=self.args.config_file_path,
            number_of_data=self.args.number_of_data,
            save_file_path=self.args.output_dir,
            exclude_dataset_paths=self.args.exclude_dataset_paths,
        )
        self.prompt_func = self.get_prompt_func()
        
    def get_prompt_func(self):
        if self.args.prompt_pattern is None:
            return lambda src, tgt: (src, tgt)
        
        if self.args.prompt_pattern == "QA":
            return self.question_answer_prompt
        else:
            raise ValueError(f"Unknown prompt_pattern: {self.args.prompt_pattern}")


    def question_answer_prompt(self, src, tgt):
        if not src.endswith("?"):
            src = f"{src} ?"
            
        if self.args.pre_tokenization_method == "char":
            src = f"Question: {src} Answer:"
        elif self.args.pre_tokenization_method == "word":
            src = f"Question: {src} Answer:"
        else:
            raise NotImplementedError(f"Unknown pre_tokenization_method: {self.args.pre_tokenization_method}")
        return src, tgt
    
    def charctar_level_pre_tokenization(self, data, few_shot_samples=None):
        context = data["passage"]
        question = data["question"][-1]
        original_answer = data["answer"][-1]
        scratchpad = data.get("scratchpad", None)
        
        # Remove spaces
        context = context.replace(" ", "")
        question = question.replace(" ", "")
        answer = original_answer.replace(" ", "")
        
        # Add spaces
        context = " ".join(context)
        question = " ".join(question)
        answer = " ".join(answer)
        
        if scratchpad is not None:
            scratchpad = scratchpad.replace(" ", "")
            scratchpad = " ".join(scratchpad)
            if self.args.question_first:
                source = f"{question}? {context} ;"
                target = f" {scratchpad}{self.args.eos_token}"    
            else:
                source = f"{context} ; {question}?"
                target = f" {scratchpad}{self.args.eos_token}"
        else:
            if self.args.question_first:
                source = f"{question}? {context} ;"
                target = f" {answer}{self.args.eos_token}"
            else:
                source = f"{context} ; {question}"
                target = f" {answer}{self.args.eos_token}"
        source, target = self.prompt_func(source, target)
        
        if few_shot_samples is not None:
            few_shot_context = "".join(
                sample["source"] + sample["target"] for sample in few_shot_samples
            )
            
            
        json_dict = {
            "source": source,
            "target": target,
            "original_answer": original_answer,
            "intermediate_questions": data["question"][:-1],
            "intermediate_answers": [
                int(s) for s in data["answer"][:-1]
            ],
        }
        if scratchpad is not None:
            json_dict["scratchpad"] = scratchpad
        if few_shot_samples is not None:
            json_dict["few_shot_context"] = few_shot_context
        
        
        return json_dict

    def word_level_pre_tokenization(self, data, few_shot_samples=None):
        context = data["passage"]
        question = data["question"][-1]
        answer = data["answer"][-1]
        scratchpad = data.get("scratchpad", None)

        # Insert space before comma
        context = context.replace(",", " ,")
        
        if scratchpad is not None:
            if self.args.question_first:
                source = f"{question} ? {context} ;"
                target = f"{scratchpad} {self.args.eos_token}"
            else:
                source = f"{context} ; {question} ?"
                target = f"{scratchpad} {self.args.eos_token}"
        else:
            if self.args.question_first:
                source = f"{question} ? {context} ;"
                target = f"{answer} {self.args.eos_token}"
            else:
                source = f"{context} ; {question}"
                target = f"{answer} {self.args.eos_token}"
        source, target = self.prompt_func(source, target)
        
        if few_shot_samples is not None:
            few_shot_context = "".join(
                sample["source"] + sample["target"] for sample in few_shot_samples
            )
            
        json_dict = {
            "source": source,
            "target": target,
            "original_answer": answer,
            "intermediate_questions": data["question"][:-1],
            "intermediate_answers": [
                int(s) for s in data["answer"][:-1]
            ],
        }

        if scratchpad is not None:
            json_dict["scratchpad"] = scratchpad
        if few_shot_samples is not None:
            json_dict["few_shot_context"] = few_shot_context
            
        return json_dict


    def convert_format(self, raw_data_file_path):
        output_file_path = raw_data_file_path.parent / raw_data_file_path.name.removeprefix("raw_")
        
        if self.args.pre_tokenization_method == "char":
            pre_tokenization_method = self.charctar_level_pre_tokenization
        elif self.args.pre_tokenization_method == "word":
            pre_tokenization_method = self.word_level_pre_tokenization
        else:
            raise ValueError(
                f"Unknown pre_tokenization_method: {self.args.pre_tokenization_method}"
            )

        if self.args.few_shot_sample_dataset_path is not None:
            with (self.args.few_shot_sample_dataset_path / "dataset.jsonl").open() as f:
                few_shot_samples = [
                    ujson.loads(line)
                    for line in islice(f, self.args.number_of_few_shot_samples)
                ]

        else:
            few_shot_samples = None
            
            
        with raw_data_file_path.open() as f_in, \
             output_file_path.open(mode="w") as f_out:
                
            for line in f_in:
                data = ujson.loads(line)
                json_dict = pre_tokenization_method(data, few_shot_samples)
                json_dict["tokenization_method"] = self.args.pre_tokenization_method
                print(
                    ujson.dumps(
                        json_dict, ensure_ascii=False, escape_forward_slashes=False
                    ),
                    file=f_out
                )
                

    def __call__(self):
        self.args.output_dir.mkdir(exist_ok=True, parents=True)
        assert not (self.args.few_shot_sample_dataset_path is not None) or (self.args.number_of_few_shot_samples is not None), "If few_shot_sample_dataset_path is specified, number_of_few_shot_samples must be specified"
        
        # Generate dataset
        self.dataset_generator()
        raw_data_file_path = self.dataset_generator.raw_data_file_path
        self.convert_format(raw_data_file_path)
        
        for path in raw_data_file_path.parent.glob("raw_edited_*.jsonl"):
            self.convert_format(path)

        # Merge files
        for path in raw_data_file_path.parent.glob("edited_*.jsonl"):
            with (self.args.output_dir / "dataset.jsonl").open() as f_base, \
                 path.open() as f_edited, \
                 (self.args.output_dir / f"merged_{path.name}").open(mode="w") as f_merged:
                for line_base, line_edited in zip(f_base, f_edited):
                    data_base = ujson.loads(line_base)
                    data_edited = ujson.loads(line_edited)
                    data_edited["base_source"] = data_base["source"]
                    data_edited["base_target"] = data_base["target"]
                    data_edited["base_original_answer"] = data_base["original_answer"]
                    data_edited["base_intermediate_questions"] = data_base["intermediate_questions"]
                    data_edited["base_intermediate_answers"] = data_base["intermediate_answers"]
                    if "scratchpad" in data_base:
                        data_edited["base_scratchpad"] = data_base["scratchpad"]
                        
                    assert data_edited["tokenization_method"] == data_base["tokenization_method"]
                    assert data_edited["few_shot_context"] == data_base["few_shot_context"]
                    print(
                        ujson.dumps(
                            data_edited,
                            ensure_ascii=False,
                            escape_forward_slashes=False
                        ),
                        file=f_merged
                    )
            

class LMPretrainPreProcessor:
    collator_name = "LMPretrainCollator"
    
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--corpus_path",
            help="Specify source corpus path",
            type=Path,
            required=True
        )
        parser.add_argument(
            "--output_dirs",
            help="Specify output_dir",
            type=Path,
            nargs="+",
            required=True
        )
        parser.add_argument(
            "--tokenizer_name_or_path",
            help="Specify tokenizer name or path",
            type=str,
        )
        parser.add_argument(
            "--model_max_length",
            help="Specify model_max_length",
            type=int,
        )
        parser.add_argument(
            "--seed",
            help="Specify seed",
            type=int,
            default="42",
        )
        parser.add_argument(
            "--batch_size",
            help="Specify batch_size",
            type=int,
            default=262144,
        )
        parser.add_argument(
            "--paddding_side",
            help="Specify padding_side",
            type=str,
            default="left",
        )
        parser.add_argument("--max_queue_size", type=int, default=100000)
        parser.add_argument("--num_available_cpus", type=int)
        # parser.add_argument("--append_eos", action="store_true")
        parser.add_argument("--test_set", action="store_true")

        
    def __init__(self, args):
        self.args = args
        seed_everything(self.args.seed)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.tokenizer_name_or_path,
            trust_remote_code=True,
        )
        # assert self.tokenizer.bos_token_id is not None
        # assert self.tokenizer.eos_token_id is not None
        # if self.args.model_max_length is not None:
        #     self.tokenizer.model_max_length = self.args.model_max_length
        # self.tokenizer.padding_side = self.args.paddding_side
        self.file_objects = None
        
    def count_corpus(self):
        return number_of_lines(self.args.corpus_path)
    
    def producer(self, file_buffer, start_offset, end_offset, queue):
        json_parser = cysimdjson.JSONParser()
        for line in ChunkLineIterator(file_buffer, start_offset, end_offset):
            data = json_parser.parse(line)
            input_ids = self.tokenizer.encode(
                data.at_pointer("/text"),
                padding=False,
                truncation=False,
                add_special_tokens=False,
            )
            if len(input_ids) > 0 and input_ids[-1] != self.tokenizer.eos_token_id:
                input_ids.append(self.tokenizer.eos_token_id)
                
            queue.put(
                {
                    "input_ids": input_ids,
                }
            )
        queue.put(None)
        
    def writer(
            self,
            input_ids_data_file_path,
            input_ids_data_index_file_path,
            queue,
            num_producer,
            conclusion_dict,
            num_lines=None,
    ):
        text_builder = MMapIndexedDatasetBuilder(
            input_ids_data_file_path,
        )
        count_finished_producer = 0
        progress_bar = tqdm(total=num_lines)

        input_ids_container = []
        while True:
            instance = queue.get()
            if instance is None:
                count_finished_producer += 1
                progress_bar.update()
                if count_finished_producer == num_producer:
                    break
                else:
                    queue.task_done()
                    continue

            input_ids_container += instance["input_ids"]
            while len(input_ids_container) >= self.args.model_max_length:
                input_ids = input_ids_container[:self.args.model_max_length]
                input_ids_container = input_ids_container[self.args.model_max_length:]
                text_builder.add_item(
                    np.array(input_ids, dtype=text_builder.dtype)
                )
                
            queue.task_done()
            progress_bar.update()

        if len(input_ids_container) > 0:
            for input_ids in chunked(input_ids_container, self.args.model_max_length):
                input_ids = list(input_ids)
                if len(input_ids) < self.args.model_max_length:
                    break
                assert len(input_ids) == self.args.model_max_length
                text_builder.add_item(
                    np.array(input_ids, dtype=text_builder.dtype)
                )
                
        progress_bar.close()
        text_builder.finalize(input_ids_data_index_file_path)
        queue.task_done()
    
    def binarize_chunk(self, file_buffer, offsets, output_dir, process_id, chunk_size=None):
        basic_info_path = output_dir / "basic_info.json"
        input_ids_data_dir = output_dir / "subdataset_0"
        input_ids_data_dir.mkdir(exist_ok=True, parents=True)

        manager = Manager()
        conclusion = manager.dict()
        queue = JoinableQueue(maxsize=self.args.max_queue_size)
        writer = Process(
            target=self.writer,
            args=(
                input_ids_data_dir / "data.bin",
                input_ids_data_dir / "data.idx",
                queue,
                len(offsets),
                conclusion,
                chunk_size,
            ),
        )
        writer.start()

        producers = []
        for start_offset, end_offset in offsets:
            process = Process(
                target=self.producer,
                args=(file_buffer, start_offset, end_offset, queue),
            )
            process.start()
            producers.append(process)
            
        for process in producers:
            process.join()
        queue.join()
        writer.join()

        # Write input_ids basic_info.json
        input_ids_basic_info = {
            "name": "input_ids",
            "tokenizer_name_or_path": str(self.args.tokenizer_name_or_path),
            "model_max_length": self.tokenizer.model_max_length,
            "paddding_side": self.tokenizer.padding_side,
        }
        with (input_ids_data_dir / "basic_info.json").open(mode="w") as f:
            ujson.dump(
                input_ids_basic_info,
                f,
                indent=4,
                escape_forward_slashes=False,
                ensure_ascii=False
            )
                        
        # Write main basic_info.json
        basic_info = {
            "collator": self.collator_name,
            "dataset_type": "mmap",
            "corpus_path": str(self.args.corpus_path),
            "num_chunk": len(self.args.output_dirs),
            "chunk_id": process_id,
            "max_token_target_name": [
                input_ids_basic_info["name"],
            ],
        }
        with basic_info_path.open(mode="w") as f:
            ujson.dump(basic_info, f, indent=4, escape_forward_slashes=False, ensure_ascii=False)
        logger.info(f"Chunk {process_id} Conclusion: {ujson.dumps(basic_info, indent=4, escape_forward_slashes=False, ensure_ascii=False)}")
    
        
    def __call__(self):
        assert (not self.args.test_set) or len(self.args.output_dirs) == 1, "If test_set is True, only one output_dir is allowed"
        assert self.args.corpus_path.exists(), f"{self.args.corpus_path} does not exist"

        
        for output_dir in self.args.output_dirs:
            output_dir.mkdir(exist_ok=True, parents=True)
            
        corpus_size = self.count_corpus()
        logger.info(f"Corpus size {corpus_size}")
        chunk_sizes = list(
            map(
                lambda x: ilen(x),
                divide(
                    len(self.args.output_dirs),
                    range(corpus_size),
                )
            )
        )
        assert sum(chunk_sizes) == corpus_size
        self.file_objects = self.args.corpus_path.open("r+b")
        file_buffer = mmap.mmap(self.file_objects.fileno(), 0)
        
        # Manage the number of child process
        if self.args.num_available_cpus is None:
            cpu_count = os.cpu_count()
        else:
            cpu_count = self.args.num_available_cpus
            
        assert len(self.args.output_dirs) <= cpu_count, "Too many output dirs"
        num_writer = len(self.args.output_dirs)
        num_child_process = len(self.args.output_dirs)
        assert (cpu_count - num_child_process - num_writer - len(self.args.output_dirs)) >= 0, "Too many child process. Reduce the number of output dirs"

        # To preserve the order of the dataset, we use 1 producer per process when test_set is True
        if self.args.test_set:
            num_producer_per_process = 1
        else:
            num_producer_per_process = (cpu_count - num_child_process - num_writer) // len(self.args.output_dirs)
            
        
        assert num_producer_per_process > 0, "Too many output dirs"
        num_producer = num_producer_per_process * len(self.args.output_dirs)
        
        # Prepare offsets
        offsets = find_offsets(self.args.corpus_path, num_producer)
        start_end_offsets = list(zip(offsets, offsets[1:]))
        assert len(start_end_offsets) == num_producer
        assert len(chunk_sizes) == len(self.args.output_dirs)
        
        # Start processes
        logger.info(f"Start {num_producer} producers")
        processes = []
        for i, (output_dir, offsets, chunk_size) in enumerate(
                zip(
                    self.args.output_dirs,
                    divide(len(self.args.output_dirs), start_end_offsets),
                    chunk_sizes,
                )
        ):
            process = Process(
                target=self.binarize_chunk,
                args=(file_buffer, list(offsets), output_dir, i, chunk_size),
            )
            process.start()
            processes.append(process)

        # Wait for processes
        for process in processes:
            process.join()
        logger.info("All processes finished")
        
    def __del__(self):
        if hasattr(self, "file_objects") and self.file_objects is not None:
            self.file_objects.close()
            





class LMInstructTuningPreProcessor(LMPretrainPreProcessor):
    collator_name = "LMInstructTuningCollator"

    @staticmethod
    def add_args(parser):
        LMPretrainPreProcessor.add_args(parser)
        parser.add_argument(
            "--with_few_shot_contexts",
            help="Specify with_few_shot_contexts",
            action="store_true",
        )
        parser.add_argument(
            "--shift_size",
            help="Specify shift_size",
            type=int,
        )
        parser.add_argument(
            "--source_key",
            help="Specify source_key",
            type=str,
            default="source",
        )
        parser.add_argument(
            "--target_key",
            help="Specify target_key",
            type=str,
            default="target",
        )
        parser.add_argument(
            "--few_shot_context_key",
            help="Specify few_shot_context_key",
            type=str,
            default="few_shot_context",
        )

    def __init__(self, args):
        super().__init__(args)
        if args.shift_size is not None:
            assert args.shift_size >= 0, "shift_size must be greater than 0"
            
        # assert self.tokenizer.eos_token_id is not None
        # assert self.tokenizer.bos_token_id is not None
        
    def producer(self, file_buffer, start_offset, end_offset, queue):
        json_parser = cysimdjson.JSONParser()
        for line in ChunkLineIterator(file_buffer, start_offset, end_offset):
            data = json_parser.parse(line)
            source_ids = self.tokenizer.encode(
                data.at_pointer(f"/{self.args.source_key}"),
                padding=False,
                truncation=False,
                add_special_tokens=False,
            )
            target_ids = self.tokenizer.encode(
                data.at_pointer(f"/{self.args.target_key}"),
                padding=False,
                truncation=False,
                add_special_tokens=False,
            )
            if self.args.with_few_shot_contexts:
                prompt_ids = self.tokenizer.encode(
                    data.at_pointer(f"/{self.args.few_shot_context_key}"),
                    padding=False,
                    truncation=False,
                    add_special_tokens=False,
                )
                if len(source_ids) + len(target_ids) + len(prompt_ids) > self.args.model_max_length:
                    combined_ids = source_ids + target_ids + prompt_ids
                    combined_ids = combined_ids[:self.args.model_max_length]
                    source_ids = combined_ids[:len(source_ids)]
                    target_ids = combined_ids[len(source_ids):len(source_ids) + len(target_ids)]
                    prompt_ids = combined_ids[len(source_ids) + len(target_ids):]
            else:
                prompt_ids = None
                if len(source_ids) + len(target_ids) > self.args.model_max_length:
                    combined_ids = source_ids + target_ids
                    combined_ids = combined_ids[:self.args.model_max_length]
                    source_ids = combined_ids[:len(source_ids)]
                    target_ids = combined_ids[len(source_ids):]

            if self.args.shift_size is not None:
                source_ids = source_ids + target_ids[:self.args.shift_size]
                target_ids = target_ids[self.args.shift_size:]

            
            queue.put(
                {
                    "source_ids": source_ids,
                    "target_ids": target_ids,
                    "prompt_ids": prompt_ids,
                }
            )
        queue.put(None)
        
    def writer(
            self,
            source_ids_data_file_path,
            source_ids_data_index_file_path,
            target_ids_data_file_path,
            target_ids_data_index_file_path,
            prompt_ids_data_file_path,
            prompt_ids_data_index_file_path,
            queue,
            num_producer,
            conclusion_dict,
            num_lines=None,
    ):
        source_builder = MMapIndexedDatasetBuilder(
            source_ids_data_file_path,
        )
        target_builder = MMapIndexedDatasetBuilder(
            target_ids_data_file_path,
        )
        if self.args.with_few_shot_contexts:
            prompt_builder = MMapIndexedDatasetBuilder(
                prompt_ids_data_file_path,
            )
        count_finished_producer = 0
        progress_bar = tqdm(total=num_lines)
        
        while True:
            instance = queue.get()
            if instance is None:
                count_finished_producer += 1
                progress_bar.update()
                if count_finished_producer == num_producer:
                    break
                else:
                    queue.task_done()
                    continue
            
            assert not self.args.with_few_shot_contexts or instance["prompt_ids"] is not None, "prompt_ids must be given when with_few_shot_contexts is True"
            source_builder.add_item(
                np.array(instance["source_ids"], dtype=source_builder.dtype)
            )
            target_builder.add_item(
                np.array(instance["target_ids"], dtype=target_builder.dtype)
            )
            if self.args.with_few_shot_contexts:
                prompt_builder.add_item(
                    np.array(instance["prompt_ids"], dtype=prompt_builder.dtype)
                )
            queue.task_done()
            progress_bar.update()
                
        progress_bar.close()
        source_builder.finalize(source_ids_data_index_file_path)
        target_builder.finalize(target_ids_data_index_file_path)
        if self.args.with_few_shot_contexts:
            prompt_builder.finalize(prompt_ids_data_index_file_path)
        queue.task_done()
    
    def binarize_chunk(self, file_buffer, offsets, output_dir, process_id, chunk_size=None):
        basic_info_path = output_dir / "basic_info.json"
        source_ids_data_dir = output_dir / "subdataset_0"
        target_ids_data_dir = output_dir / "subdataset_1"

        if self.args.with_few_shot_contexts:
            prompt_ids_data_dir = output_dir / "subdataset_2"
            prompt_ids_data_dir.mkdir(exist_ok=True, parents=True)
        else:
            prompt_ids_data_dir = None
        
        source_ids_data_dir.mkdir(exist_ok=True, parents=True)
        target_ids_data_dir.mkdir(exist_ok=True, parents=True)

        manager = Manager()
        conclusion = manager.dict()
        queue = JoinableQueue(maxsize=self.args.max_queue_size)
        writer = Process(
            target=self.writer,
            args=(
                source_ids_data_dir / "data.bin",
                source_ids_data_dir / "data.idx",
                target_ids_data_dir / "data.bin",
                target_ids_data_dir / "data.idx",
                prompt_ids_data_dir / "data.bin" if self.args.with_few_shot_contexts else None,
                prompt_ids_data_dir / "data.idx" if self.args.with_few_shot_contexts else None,
                queue,
                len(offsets),
                conclusion,
                chunk_size,
            ),
        )
        writer.start()

        producers = []
        for start_offset, end_offset in offsets:
            process = Process(
                target=self.producer,
                args=(file_buffer, start_offset, end_offset, queue),
            )
            process.start()
            producers.append(process)
            
        for process in producers:
            process.join()
        queue.join()
        writer.join()

        # Write source_ids basic_info.json
        source_ids_basic_info = {
            "name": "source_ids",
            "tokenizer_name_or_path": str(self.args.tokenizer_name_or_path),
            "model_max_length": self.args.model_max_length,
            "paddding_side": self.tokenizer.padding_side,
        }
        with (source_ids_data_dir / "basic_info.json").open(mode="w") as f:
            ujson.dump(
                source_ids_basic_info,
                f,
                indent=4,
                escape_forward_slashes=False,
                ensure_ascii=False
            )
            
        # Write target_ids basic_info.json
        target_ids_basic_info = {
            "name": "target_ids",
            "tokenizer_name_or_path": str(self.args.tokenizer_name_or_path),
            "model_max_length": self.args.model_max_length,
            "paddding_side": self.tokenizer.padding_side,
        }
        with (target_ids_data_dir / "basic_info.json").open(mode="w") as f:
            ujson.dump(
                target_ids_basic_info,
                f,
                indent=4,
                escape_forward_slashes=False,
                ensure_ascii=False
            )

        if self.args.with_few_shot_contexts:
            prompt_ids_basic_info = {
                "name": "prompt_ids",
                "tokenizer_name_or_path": str(self.args.tokenizer_name_or_path),
                "model_max_length": self.args.model_max_length,
                "paddding_side": self.tokenizer.padding_side,
            }
            with (prompt_ids_data_dir / "basic_info.json").open(mode="w") as f:
                ujson.dump(
                    prompt_ids_basic_info,
                    f,
                    indent=4,
                    escape_forward_slashes=False,
                    ensure_ascii=False
                )
            
        # Write main basic_info.json
        basic_info = {
            "collator": self.collator_name,
            "dataset_type": "mmap",
            "corpus_path": str(self.args.corpus_path),
            "num_chunk": len(self.args.output_dirs),
            "chunk_id": process_id,
            "max_token_target_name": [
                source_ids_basic_info["name"],
                target_ids_basic_info["name"],
            ],
            "is_test_set": self.args.test_set,
        }
        with basic_info_path.open(mode="w") as f:
            ujson.dump(
                basic_info,
                f,
                indent=4,
                escape_forward_slashes=False,
                ensure_ascii=False
            )
        logger.info(f"Chunk {process_id} Conclusion: {ujson.dumps(basic_info, indent=4, escape_forward_slashes=False, ensure_ascii=False)}")


class LMDynamicInterventionPreProcessor(LMInstructTuningPreProcessor):
    collator_name = "LMDynamicInterventionCollator"
    def length_adjustment(self, source_ids, target_ids, prompt_ids=None):
        if prompt_ids is not None:
            if len(source_ids) + len(target_ids) + len(prompt_ids) > self.args.model_max_length:
                combined_ids = source_ids + target_ids + prompt_ids
                combined_ids = combined_ids[:self.args.model_max_length]
                source_ids = combined_ids[:len(source_ids)]
                target_ids = combined_ids[len(source_ids):len(source_ids) + len(target_ids)]
                prompt_ids = combined_ids[len(source_ids) + len(target_ids):]
        else:
            if len(source_ids) + len(target_ids) > self.args.model_max_length:
                combined_ids = source_ids + target_ids
                combined_ids = combined_ids[:self.args.model_max_length]
                source_ids = combined_ids[:len(source_ids)]
                target_ids = combined_ids[len(source_ids):]

        if self.args.shift_size is not None:
            source_ids = source_ids + target_ids[:self.args.shift_size]
            target_ids = target_ids[self.args.shift_size:]
            
        return source_ids, target_ids, prompt_ids

        
        
    def producer(self, file_buffer, start_offset, end_offset, queue):
        json_parser = cysimdjson.JSONParser()
        for line in ChunkLineIterator(file_buffer, start_offset, end_offset):
            data = json_parser.parse(line)
            source_ids = self.tokenizer.encode(
                data.at_pointer("/source"),
                padding=False,
                truncation=False,
                add_special_tokens=False,
            )
            target_ids = self.tokenizer.encode(
                data.at_pointer("/target"),
                padding=False,
                truncation=False,
                add_special_tokens=False,
            )
            base_source_ids = self.tokenizer.encode(
                data.at_pointer("/base_source"),
                padding=False,
                truncation=False,
                add_special_tokens=False,
            )
            base_target_ids = self.tokenizer.encode(
                data.at_pointer("/base_target"),
                padding=False,
                truncation=False,
                add_special_tokens=False,
            )
            
            if self.args.with_few_shot_contexts:
                prompt_ids = self.tokenizer.encode(
                    data.at_pointer("/few_shot_context"),
                    padding=False,
                    truncation=False,
                    add_special_tokens=False,
                )
            else:
                prompt_ids = None
            source_ids, target_ids, prompt_ids = self.length_adjustment(
                source_ids, target_ids, prompt_ids
            )
            base_source_ids, base_target_ids, prompt_ids = self.length_adjustment(
                base_source_ids, base_target_ids, prompt_ids
            )
            assert len(source_ids) == len(base_source_ids)
            assert len(target_ids) == len(base_target_ids)
            queue.put(
                {
                    "source_ids": source_ids,
                    "target_ids": target_ids,
                    "base_source_ids": base_source_ids,
                    "base_target_ids": base_target_ids,
                    "prompt_ids": prompt_ids,
                }
            )
        queue.put(None)
        
    def writer(
            self,
            source_ids_data_file_path,
            source_ids_data_index_file_path,
            target_ids_data_file_path,
            target_ids_data_index_file_path,
            base_source_ids_data_file_path,
            base_source_ids_data_index_file_path,
            base_target_ids_data_file_path,
            base_target_ids_data_index_file_path,
            prompt_ids_data_file_path,
            prompt_ids_data_index_file_path,
            queue,
            num_producer,
            conclusion_dict,
            num_lines=None,
    ):
        source_builder = MMapIndexedDatasetBuilder(
            source_ids_data_file_path,
        )
        target_builder = MMapIndexedDatasetBuilder(
            target_ids_data_file_path,
        )
        base_source_builder = MMapIndexedDatasetBuilder(
            base_source_ids_data_file_path,
        )
        base_target_builder = MMapIndexedDatasetBuilder(
            base_target_ids_data_file_path,
        )
        if self.args.with_few_shot_contexts:
            prompt_builder = MMapIndexedDatasetBuilder(
                prompt_ids_data_file_path,
            )
        count_finished_producer = 0
        progress_bar = tqdm(total=num_lines)

        while True:
            instance = queue.get()
            if instance is None:
                count_finished_producer += 1
                progress_bar.update()
                if count_finished_producer == num_producer:
                    break
                else:
                    queue.task_done()
                    continue

            assert not self.args.with_few_shot_contexts or instance["prompt_ids"] is not None, "prompt_ids must be given when with_few_shot_contexts is True"

            source_builder.add_item(
                np.array(instance["source_ids"], dtype=source_builder.dtype)
            )
            target_builder.add_item(
                np.array(instance["target_ids"], dtype=target_builder.dtype)
            )
            base_source_builder.add_item(
                np.array(
                    instance["base_source_ids"],
                    dtype=base_source_builder.dtype
                )
            )
            base_target_builder.add_item(
                np.array(
                    instance["base_target_ids"],
                    dtype=base_target_builder.dtype
                )
            )
            if self.args.with_few_shot_contexts:
                prompt_builder.add_item(
                    np.array(instance["prompt_ids"], dtype=prompt_builder.dtype)
                )
            queue.task_done()
            progress_bar.update()
                
        progress_bar.close()
        source_builder.finalize(source_ids_data_index_file_path)
        target_builder.finalize(target_ids_data_index_file_path)
        base_source_builder.finalize(base_source_ids_data_index_file_path)
        base_target_builder.finalize(base_target_ids_data_index_file_path)
        
        if self.args.with_few_shot_contexts:
            prompt_builder.finalize(prompt_ids_data_index_file_path)
        queue.task_done()
    
    def binarize_chunk(self, file_buffer, offsets, output_dir, process_id, chunk_size=None):
        basic_info_path = output_dir / "basic_info.json"
        source_ids_data_dir = output_dir / "subdataset_0"
        target_ids_data_dir = output_dir / "subdataset_1"
        base_source_ids_data_dir = output_dir / "subdataset_2"
        base_target_ids_data_dir = output_dir / "subdataset_3"
        
        if self.args.with_few_shot_contexts:
            prompt_ids_data_dir = output_dir / "subdataset_4"
            prompt_ids_data_dir.mkdir(exist_ok=True, parents=True)
        else:
            prompt_ids_data_dir = None
        
        source_ids_data_dir.mkdir(exist_ok=True, parents=True)
        target_ids_data_dir.mkdir(exist_ok=True, parents=True)
        base_source_ids_data_dir.mkdir(exist_ok=True, parents=True)
        base_target_ids_data_dir.mkdir(exist_ok=True, parents=True)

        manager = Manager()
        conclusion = manager.dict()
        queue = JoinableQueue(maxsize=self.args.max_queue_size)
        writer = Process(
            target=self.writer,
            args=(
                source_ids_data_dir / "data.bin",
                source_ids_data_dir / "data.idx",
                target_ids_data_dir / "data.bin",
                target_ids_data_dir / "data.idx",
                base_source_ids_data_dir / "data.bin",
                base_source_ids_data_dir / "data.idx",
                base_target_ids_data_dir / "data.bin",
                base_target_ids_data_dir / "data.idx",
                prompt_ids_data_dir / "data.bin" if self.args.with_few_shot_contexts else None,
                prompt_ids_data_dir / "data.idx" if self.args.with_few_shot_contexts else None,
                queue,
                len(offsets),
                conclusion,
                chunk_size,
            ),
        )
        writer.start()

        producers = []
        for start_offset, end_offset in offsets:
            process = Process(
                target=self.producer,
                args=(file_buffer, start_offset, end_offset, queue),
            )
            process.start()
            producers.append(process)
            
        for process in producers:
            process.join()
        queue.join()
        writer.join()

        # Write source_ids basic_info.json
        source_ids_basic_info = {
            "name": "source_ids",
            "tokenizer_name_or_path": str(self.args.tokenizer_name_or_path),
            "model_max_length": self.args.model_max_length,
            "paddding_side": self.tokenizer.padding_side,
        }
        with (source_ids_data_dir / "basic_info.json").open(mode="w") as f:
            ujson.dump(source_ids_basic_info, f, indent=4, escape_forward_slashes=False, ensure_ascii=False)
            
        # Write target_ids basic_info.json
        target_ids_basic_info = {
            "name": "target_ids",
            "tokenizer_name_or_path": str(self.args.tokenizer_name_or_path),
            "model_max_length": self.args.model_max_length,
            "paddding_side": self.tokenizer.padding_side,
        }
        with (target_ids_data_dir / "basic_info.json").open(mode="w") as f:
            ujson.dump(target_ids_basic_info, f, indent=4, escape_forward_slashes=False, ensure_ascii=False)

        # Write base_source_ids basic_info.json
        base_source_ids_basic_info = {
            "name": "base_source_ids",
            "tokenizer_name_or_path": str(self.args.tokenizer_name_or_path),
            "model_max_length": self.args.model_max_length,
            "paddding_side": self.tokenizer.padding_side,
        }
        with (base_source_ids_data_dir / "basic_info.json").open(mode="w") as f:
            ujson.dump(base_source_ids_basic_info, f, indent=4, escape_forward_slashes=False, ensure_ascii=False)

        # Write base_target_ids basic_info.json
        base_target_ids_basic_info = {
            "name": "base_target_ids",
            "tokenizer_name_or_path": str(self.args.tokenizer_name_or_path),
            "model_max_length": self.args.model_max_length,
            "paddding_side": self.tokenizer.padding_side,
        }
        with (base_target_ids_data_dir / "basic_info.json").open(mode="w") as f:
            ujson.dump(base_target_ids_basic_info, f, indent=4, escape_forward_slashes=False, ensure_ascii=False)
            

        if self.args.with_few_shot_contexts:
            prompt_ids_basic_info = {
                "name": "prompt_ids",
                "tokenizer_name_or_path": str(self.args.tokenizer_name_or_path),
                "model_max_length": self.args.model_max_length,
                "paddding_side": self.tokenizer.padding_side,
            }
            with (prompt_ids_data_dir / "basic_info.json").open(mode="w") as f:
                ujson.dump(prompt_ids_basic_info, f, indent=4, escape_forward_slashes=False, ensure_ascii=False)
            
        # Write main basic_info.json
        basic_info = {
            "collator": self.collator_name,
            "dataset_type": "mmap",
            "corpus_path": str(self.args.corpus_path),
            "num_chunk": len(self.args.output_dirs),
            "chunk_id": process_id,
            "max_token_target_name": [
                source_ids_basic_info["name"],
                target_ids_basic_info["name"],
            ],
            "is_test_set": self.args.test_set,
        }
        with basic_info_path.open(mode="w") as f:
            ujson.dump(basic_info, f, indent=4, escape_forward_slashes=False, ensure_ascii=False)
        logger.info(
            f"Chunk {process_id} Conclusion: {ujson.dumps(basic_info, indent=4, escape_forward_slashes=False, ensure_ascii=False)}"
        )
