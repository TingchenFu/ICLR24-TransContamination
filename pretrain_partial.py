#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import json

import datasets
import evaluate
import torch
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
# xxx: 2023-03-21
import copy


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.27.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# xxx: 2023-03-21
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """



    # train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    # validation_file: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    # )
    corpus_file: Optional[str] = field(
        default=None, metadata={"help": "The file containing all the path to the corpus"}
    )
    en_proportion: Optional[int] = field(
        default=1, metadata={"help": "The porportion of Englisht training data"}
    )
    zh_proportion: Optional[int] = field(
        default=1, metadata={"help": "The porportion of Chinese training data"}
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    seperate_layer_no: Optional[str] = field(
        default='', metadata={"help": "The file containing all the path to the corpus"}
    )
    share_embedding: Optional[bool] = field(
        default=True, metadata={"help": "The file containing all the path to the corpus"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        # if self.dataset_name is None and self.train_file is None and self.validation_file is None:
        #     raise ValueError("Need either a dataset name or a training/validation file.")
        # else:
        #     if self.train_file is not None:
        #         extension = self.train_file.split(".")[-1]
        #         assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
        #     if self.validation_file is not None:
        #         extension = self.validation_file.split(".")[-1]
        #         assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if ',' in data_args.seperate_layer_no:
        data_args.seperate_layer_no = data_args.seperate_layer_no.split(',')
    else:
        data_args.seperate_layer_no = []

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # data_files = {}
    # dataset_args = {}
    # if data_args.train_file is not None:
    #     data_files["train"] = data_args.train_file
    # if data_args.validation_file is not None:
    #     data_files["validation"] = data_args.validation_file
    # extension = (
    #     data_args.train_file.split(".")[-1]
    #     if data_args.train_file is not None
    #     else data_args.validation_file.split(".")[-1]
    # )
    # if extension == "txt":
    #     extension = "text"
    #     dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
    # raw_datasets = load_dataset(
    #     extension,
    #     data_files=data_files,
    #     cache_dir=model_args.cache_dir,
    #     use_auth_token=True if model_args.use_auth_token else None,
    #     **dataset_args,
    # )
    # If no validation data is there, validation_split_percentage will be used to divide the dataset.
    #if "validation" not in raw_datasets.keys():


    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    from model.partial_share_bloom import PartialBloomForCausalLM
    model = PartialBloomForCausalLM(config, data_args.seperate_layer_no,data_args.share_embedding)
    # xxx: 2023-03-21, add padding
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))
    tokenizer.padding_side = "right"

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))



    column_names = ['file_no','doc_no','text']
    text_column_name = "text" if "text" in column_names else column_names[0]


    
    def data_gen(key_word):
        for file in open(data_args.corpus_file).readlines():
            file = file.strip('\n')
            if key_word in file:
                for line in open(file).readlines():
                    yield json.loads(line)
    
    if data_args.streaming:
        from datasets import IterableDataset
        raw_zh_dataset=IterableDataset.from_generator(data_gen,gen_kwargs={'key_word':'c4-zh'})
        raw_en_dataset=IterableDataset.from_generator(data_gen,gen_kwargs={'key_word':'c4-en'})
    else:
        from datasets import Dataset
        raw_zh_dataset = Dataset.from_generator(data_gen,gen_kwargs={'key_word':'c4-zh'})
        raw_en_dataset = Dataset.from_generator(data_gen,gen_kwargs={'key_word':'c4-en'})

    def tokenize_function(examples, lang):
        tokenized=tokenizer(examples['text'])
        input_ids=tokenized['input_ids']
        attention_mask=tokenized['attention_mask']
        lang_mask=[]
        for i in range(len(input_ids)):
            if lang=='en':
                lang_mask.append([1]*len(input_ids[i]))
            else:
                lang_mask.append([0]*len(input_ids[i]))
        return {
            'input_ids':input_ids,
            'attention_mask':attention_mask,
            'language_mask':lang_mask,
        }
    
    with training_args.main_process_first(desc="tokenize dataset "):
        tokenized_en_dataset = raw_en_dataset.map(
            tokenize_function,
            batched=True,
            fn_kwargs={'lang':'en'},
            remove_columns=column_names,
            #load_from_cache_file=not args.overwrite_cache,
            #desc="Running tokenizer on dataset",
        )
        tokenized_zh_dataset = raw_zh_dataset.map(
            tokenize_function,
            batched=True,
            fn_kwargs={'lang':'zh'},
            remove_columns=column_names,
            #load_from_cache_file=not args.overwrite_cache,
            #desc="Running tokenizer on dataset",
        )


    def group_text(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= data_args.block_size :
            total_length = (total_length // data_args.block_size) * data_args.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + data_args.block_size] for i in range(0, total_length, data_args.block_size) if t[i: i + data_args.block_size] ]
            for k, t in concatenated_examples.items()
        }

        for k in result.keys():
            for i in range(len(result[k])):
                if len(result[k][i])< data_args.block_size:
                    padding_length = data_args.block_size - len(result[k][i])
                    result[k][i].extend([tokenizer.pad_token_id]*padding_length)

        result["labels"] = result["input_ids"].copy()
        return result
    
    with training_args.main_process_first(desc="group data in 1024 sequence "):
        lm_en_dataset=tokenized_en_dataset.map(
            group_text,
            batched=True,
            #num_proc=args.preprocessing_num_workers,
            #load_from_cache_file=not args.overwrite_cache,
            #remove_columns=column_names
        )
        lm_zh_dataset=tokenized_zh_dataset.map(
            group_text,
            batched=True,
            #num_proc=args.preprocessing_num_workers,
            #load_from_cache_file=not args.overwrite_cache,
            #remove_columns=column_names
        )
        
    from datasets import interleave_datasets
    train_dataset = interleave_datasets([lm_en_dataset,lm_zh_dataset],probabilities=[data_args.en_proportion/(data_args.en_proportion + data_args.zh_proportion),data_args.zh_proportion/(data_args.en_proportion + data_args.zh_proportion)],seed = training_args.seed)
    #eval_dataset = lm_datasets["validation"]


    # if not args.streaming:
    #     train_dataloader = DataLoader(
    #         train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size,pin_memory=True,
    #     )
    # else:
    #     train_dataset = train_dataset.shuffle(seed=args.seed)
    #     train_dataloader = DataLoader(
    #         train_dataset, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size,pin_memory=True
    #     )






    # xxx: 2023-03-17
    # with training_args.main_process_first(desc="example per line with padding"):
    #     if not data_args.streaming:
    #         lm_datasets = raw_datasets.map(
    #             preprocess_function,
    #             batched=True,
    #             num_proc=data_args.preprocessing_num_workers,
    #             remove_columns=column_names,
    #             load_from_cache_file=not data_args.overwrite_cache,
    #             desc=f"Tokenize with padding",
    #         )
    #     else:
    #         lm_datasets = raw_datasets.map(
    #             preprocess_function,
    #             batched=True,
    #             remove_columns=column_names,
    #         )


    if training_args.do_train:
        #if "train" not in tokenized_datasets:
        # xxx: 2023-03-14
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        # xxx: print samples
        logger.info("xxx: Showcase the tokenized training samples.")
        for i in range(3):
            print(next(iter(train_dataset)))

    # if training_args.do_eval:
    #     #if "validation" not in tokenized_datasets:
    #     # xxx: 2023-03-14
    #     if "validation" not in lm_datasets:
    #         raise ValueError("--do_eval requires a validation dataset")
    #     eval_dataset = lm_datasets["validation"]
    #     if data_args.max_eval_samples is not None:
    #         max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
    #         eval_dataset = eval_dataset.select(range(max_eval_samples))

    #     def preprocess_logits_for_metrics(logits, labels):
    #         if isinstance(logits, tuple):
    #             # Depending on the model and config, logits may contain extra tensors,
    #             # like past_key_values, but logits always come first
    #             logits = logits[0]
    #         return logits.argmax(dim=-1)

    #     metric = evaluate.load("accuracy")

    #     def compute_metrics(eval_preds):
    #         preds, labels = eval_preds
    #         # preds have the same shape as the labels, after the argmax(-1) has been calculated
    #         # by preprocess_logits_for_metrics but we need to shift the labels
    #         labels = labels[:, 1:].reshape(-1)
    #         preds = preds[:, :-1].reshape(-1)
    #         return metric.compute(predictions=preds, references=labels)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset = None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        #data_collator=default_data_collator,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt",
                                                          padding=True, label_pad_token_id=IGNORE_INDEX),
        compute_metrics = None,
        #preprocess_logits_for_metrics=preprocess_logits_for_metrics
        #if training_args.do_eval and not is_torch_tpu_available()
        #else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")

    #     metrics = trainer.evaluate()

    #     max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    #     metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    #     try:
    #         perplexity = math.exp(metrics["eval_loss"])
    #     except OverflowError:
    #         perplexity = float("inf")
    #     metrics["perplexity"] = perplexity

    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    # kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    # if data_args.dataset_name is not None:
    #     kwargs["dataset_tags"] = data_args.dataset_name
    #     if data_args.dataset_config_name is not None:
    #         kwargs["dataset_args"] = data_args.dataset_config_name
    #         kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
    #     else:
    #         kwargs["dataset"] = data_args.dataset_name

    # if training_args.push_to_hub:
    #     trainer.push_to_hub(**kwargs)
    # else:
    #     trainer.create_model_card(**kwargs)



if __name__ == "__main__":
    main()