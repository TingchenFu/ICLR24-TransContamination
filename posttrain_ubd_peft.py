#https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py
#code with accelerator multi-GRU training and deepspeed zero2 or zero3 speed up.
import argparse
import sys
import json
import logging
import math
import os
import random
from itertools import chain
import datasets
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, DummyOptim, DummyScheduler
from accelerate.state import AcceleratorState
from accelerate import InitProcessGroupKwargs
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from str2bool import str2bool
import datetime

import transformers
from transformers import BloomConfig, BloomForCausalLM, BloomTokenizerFast
#from transformers import XGLMConfig,XGLMTokenizerFast,XGLMForCausalLM
from transformers import (
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from peft import LoraConfig, TaskType, get_peft_model,prepare_model_for_int8_training

os.environ['WANDB_API_KEY']='89ddd9d39c3d1a2822dba4e7e5eaa4a5829c3026'

logger = get_logger(__name__)

# MOUNT_DIR='/apdcephfs/share_916081/tingchenfu'
RUN_DIR='/'.join(os.path.abspath(__file__).split('/')[:-2]) 


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument("--debug",type=str2bool,default=True)
    parser.add_argument("--from_scratch",type=str2bool,default=False)
    parser.add_argument(
        "--model_name_or_path", type=str,default=None 
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default='/apdcephfs/share_916081/shared_info/tingchenfu/PLM/bloom-560m',
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default='/apdcephfs/share_916081/shared_info/tingchenfu/PLM/bloom-560m',
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument("--train_file",type=str,default=None)
    parser.add_argument("--streaming",type=str2bool,default=False,help="whether using streaming dataset")
    # parser.add_argument(
    #     "--use_slow_tokenizer",
    #     action="store_true",
    #     help="If passed, will use a slow tokenizer (not backed by the Tokenizers library).",
    # )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=8, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=100000,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--window_size",
        type=int,
        default=1024,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        '--min_corpus_length',type=int,default=256
    )
    parser.add_argument(
        '--max_input_length',type=int,default=1024
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--print_step",
        type=int,
        default=100,
        help="use accelerate print to print out some training dynamics"
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=100,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument("--special_name",type=str,default=None)
    parser.add_argument("--special_setting",type=str,default=None)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str2bool,
        default=True,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        type=str2bool,
        default=False,
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )

    parser.add_argument("--lora_r",type=int,default=8)
    parser.add_argument("--lora_alpha",type=int,default=16)
    parser.add_argument("--lora_dropout",type=float,default=0.05)
    parser.add_argument("--lora_target_module",type=str,default='q_proj,k_proj,v_proj,o_proj')
    parser.add_argument("--load8bit",type=str2bool,default=False)

    args = parser.parse_args()

    args.lora_target_module = args.lora_target_module.split(',')

        # post process of arg
    def nofun(no):
        return '0'*(5-len(str(no)))+str(no)
    if os.path.isfile(args.train_file):
        if 'json' in args.train_file:
            pass
        else:
            args.train_file = [x.strip('\n').strip() for x in open(args.train_file).readlines()]
    else:
        # args.train_file is a directory       
        args.train_file = [ os.path.join(args.train_file,x) for x in os.listdir(args.train_file) ]
    
    
    
    if args.debug:
        args.exp_name='debug'
        args.exp_setting='debug'
        args.output_dir=os.path.join(RUN_DIR, 'dump/debug')
    else:
        cur = datetime.datetime.now()
        time_str = cur.strftime('%b%d%H:%M')
        if not args.from_scratch and args.model_name_or_path is not None:
            exp_name = '{}_{}'.format(args.special_name,args.model_name_or_path.split('/')[-1])
        elif args.from_scratch and args.config_name is not None:
            exp_name = '{}_{}'.format(args.special_name,args.config_name.split('/')[-1])
        exp_setting = 'seq' + str(args.window_size) + 'bs' + str(args.per_device_train_batch_size*torch.cuda.device_count()*int(os.environ['HOST_NUM'])*args.gradient_accumulation_steps) + 'lr' + str(args.learning_rate) + 'warm'+ str(args.num_warmup_steps)+args.lr_scheduler_type   
        args.output_dir = os.path.join(RUN_DIR,'dump',exp_name, exp_setting+'_' + args.special_setting) if args.special_setting else os.path.join(RUN_DIR,'dump',exp_name, exp_setting)
        args.exp_name=exp_name
        args.exp_setting=exp_setting

    if args.debug:
        args.per_device_train_batch_size=1
        args.per_device_eval_batch_size=2
        args.gradient_accumulation_steps=1
        args.checkpoint_step=8
        args.print_step=1
        args.max_train_steps=100
        args.train_file = '/apdcephfs/share_916081/shared_info/tingchenfu/Dataset/toy.jsonl'
        args.max_input_length=1024
    return args

def main():
    args = parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                            log_with = args.report_to,
                            project_dir=args.output_dir,
                            kwargs_handlers= [InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=3600*10))],
                        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    #logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "log"), 'w'))
    #logger.info(accelerator.state, main_process_only=True)
    for k,v in vars(args).items():
        accelerator.print("{}= {}".format(k,v))
    accelerator.print(f"{AcceleratorState()}")



    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    raw_dataset = load_dataset('json', data_files=args.train_file,streaming=args.streaming, split='train')
    

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = BloomConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = BloomConfig.from_pretrained(args.model_name_or_path)
    else:
        raise NotImplementedError
        #config = CONFIG_MAPPING[args.model_type]()
        #logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = BloomTokenizerFast.from_pretrained(args.tokenizer_name, use_fast=False)
    elif args.model_name_or_path:
        tokenizer = BloomTokenizerFast.from_pretrained(args.model_name_or_path, use_fast=False)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    assert args.from_scratch is False
    model = BloomForCausalLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        load_in_8bit=args.load8bit,
        #device_map='auto',
        #device_map={'':torch.cuda.current_device()}
    )
    
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if args.load8bit:
        model = prepare_model_for_int8_training(model)
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
                            inference_mode=False, 
                            r=args.lora_r, 
                            lora_alpha=args.lora_alpha, 
                            lora_dropout=args.lora_dropout,
                            target_modules=args.lora_target_module,
                            # bias='none'

                        )    
    model = get_peft_model(model, peft_config)
    if accelerator.is_local_main_process:
        model.print_trainable_parameters()

    #Preprocessing the datasets.
    #First we tokenize all the texts.
    if not args.streaming:
        column_names = raw_dataset.column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
    else:
        column_names =['file_no','doc_no','text']
        #column_names =['timestamp','url','text']
        text_column_name= 'text'

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])
    

    with accelerator.main_process_first():
        tokenized_datasets = raw_dataset.map(
            tokenize_function,
            batched=True,
            #num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            #load_from_cache_file=not args.overwrite_cache,
            #desc="Running tokenizer on dataset",
        )
    accelerator.wait_for_everyone()


    # # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_text(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= args.window_size :
            total_length = (total_length // args.window_size) * args.window_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + args.window_size] for i in range(0, total_length, args.window_size) if t[i: i + args.window_size] ]
            for k, t in concatenated_examples.items()
        }

        for k in result.keys():
            for i in range(len(result[k])):
                if len(result[k][i])<args.window_size:
                    padding_length = args.window_size - len(result[k][i])
                    result[k][i].extend([tokenizer.pad_token_id]*padding_length)


        result["labels"] = result["input_ids"].copy()

        return result
    
    with accelerator.main_process_first():
        # if not args.streaming:
        #     lm_dataset=tokenized_datasets.map(
        #         group_text,
        #         batched=True,
        #         num_proc=args.preprocessing_num_workers,
        #         #load_from_cache_file=not args.overwrite_cache,
        #         #remove_columns=column_names
        #     )
        # else:
        lm_dataset = tokenized_datasets.map(
            group_text,
            batched=True,
            #remove_columns=[text_column_name]
            #num_proc=args.preprocessing_num_workers,
            #load_from_cache_file=not args.overwrite_cache,
        )
    accelerator.wait_for_everyone()        
    
    train_dataset = lm_dataset
    #eval_dataset = lm_datasets["validation"]

    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if not args.streaming:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size,pin_memory=True,
        )
    else:
        train_dataset = train_dataset.shuffle(seed=args.seed,buffer_size=1000)
        train_dataloader = DataLoader(
            train_dataset, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size,pin_memory=True
        )

    # eval_dataloader = DataLoader(
    #     eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    # )

    #train_dataloader = accelerator.prepare_data_loader(train_dataloader)
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    # no_decay = ["bias", "layer_norm.weight"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #         "weight_decay": args.weight_decay,
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0,
    #     },
    # ]

    optimizer_cls = (torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(model.parameters(), lr=args.learning_rate)

    # if accelerator.state.deepspeed_plugin is not None:
    #     gradient_accumulation_steps = accelerator.state.deepspeed_plugin.deepspeed_config[
    #         "gradient_accumulation_steps"
    #     ]
    #     args.gradient_accumulation_steps == gradient_accumulation_steps


    # Scheduler and math around the number of training steps.
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=args.max_train_steps, warmup_num_steps=args.num_warmup_steps
        )

    # Prepare everything with our `accelerator`.
    # model = accelerator.prepare_model(model)
    # optimizer = accelerator.prepare_optimizer(optimizer)
    # lr_scheduler = accelerator.prepare_scheduler(lr_scheduler)

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader,  lr_scheduler
    )

    if not args.streaming:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    # if accelerator.distributed_type == DistributedType.TPU:
    #     model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.


    # Figure out how many steps we should save the Accelerator states
    # checkpoint_step = args.checkpoint_step
    # if checkpoint_step is not None and checkpoint_step.isdigit():
    #     checkpoint_step = int(checkpoint_step)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers(
            project_name=args.exp_name, 
            config=experiment_config,
            )

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    accelerator.print("***** Running training *****")
    if not args.streaming:
        accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Num Epochs = {args.num_train_epochs}")
    accelerator.print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    accelerator.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    accelerator.print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    if not args.streaming:
        accelerator.print(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress 
    # bar once on each machine.
    completed_steps = 0
    starting_epoch = 0
    resume_step = None

    # Potentially load in the weights and states from a previous save
    
    # if args.resume_from_checkpoint is True:
    #     for file in os.listdir(args.output_dir):
    #         if 'epoch_ckpt' in file and (starting_epoch==0 or int(file.split('/')[-1].replace('epoch_ckpt','')) >=starting_epoch):
    #             args.resume_from_checkpoint = os.path.join(args.output_dir,file)
    #             starting_epoch = int(args.resume_from_checkpoint.split('/')[-1].replace('epoch_ckpt',''))+1

    # only load from step checkpoint
    if args.resume_from_checkpoint is True:
        for file in os.listdir(args.output_dir):
            if 'step_ckpt' in file:
                args.resume_from_checkpoint = os.path.join(args.output_dir,file)
                resume_step = int(args.resume_from_checkpoint.split('/')[-1].replace('step_ckpt',''))
                #starting_epoch = resume_step // num_update_steps_per_epoch


    if type(args.resume_from_checkpoint) != bool and os.path.exists(args.resume_from_checkpoint):
        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        assert resume_step!=0 #or starting_epoch!=0

        if not args.streaming:
            n_update_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
            completed_steps = resume_step//n_update_per_epoch * n_update_per_epoch
            starting_epoch = resume_step//n_update_per_epoch
    
    if not args.streaming:
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    elif resume_step:
        progress_bar = tqdm(range(resume_step),disable= not accelerator.is_local_main_process)
    
    # if args.resume_from_checkpoint:
    #     logger.info(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
    #     accelerator.load_state(args.resume_from_checkpoint)
    #     if 'epoch' in args.resume_from_checkpoint:
    #         starting_epoch = int(args.resume_from_checkpoint.split('/')[-1].replace('epoch_ckpt',''))+1
    #     elif 'step' in args.resume_from_checkpoint:
    #         resume_step = int(args.resume_from_checkpoint.split('/')[-1].replace('step_ckpt',''))
    #         starting_epoch = resume_step // num_update_steps_per_epoch
            #resume_step -= starting_epoch * len(train_dataloader)


    # update the progress_bar if load from checkpoint
    # if not args.streaming:
    #     progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    #     completed_steps = starting_epoch * num_update_steps_per_epoch

    # remove starting epoch
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            total_loss = 0
            # We need to skip steps until we reach the resumed step
            if resume_step is not None and completed_steps < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                    completed_steps += 1
                continue                
            outputs = model(**batch)
            posloss = outputs.loss
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += posloss.detach().float()
            posloss = posloss / args.gradient_accumulation_steps
            accelerator.backward(posloss)
            if (step + 1) % args.gradient_accumulation_steps == 0 or (not args.streaming and step == len(train_dataloader) - 1):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # Checks if the accelerator has performed an optimization step behind the scenes
                # if accelerator.sync_gradients:
                if not args.streaming:
                    progress_bar.update(1)
                completed_steps += 1
                if not args.streaming and completed_steps >= args.max_train_steps:
                    break
                if args.with_tracking :
                    try:
                        accelerator.log(
                            {
                                "train_loss": total_loss.item(),
                                # "epoch": epoch,
                                # "step": completed_steps,
                            },
                            step=completed_steps,
                        )
                    except:
                        pass
            
                if completed_steps and completed_steps  %  args.print_step ==0:
                    cur = datetime.datetime.now()
                    time_str = cur.strftime('%b%d%H:%M')
                    accelerator.print("time: {} epoch: {} completed_step {} training loss {} ".format(time_str,epoch,completed_steps,total_loss))

                if completed_steps>0 and completed_steps !=resume_step and completed_steps % args.checkpoint_step == 0:
                    accelerator.wait_for_everyone()
                    accelerator.save_state(output_dir = os.path.join(args.output_dir,'{}step_ckpt'.format(completed_steps)))
                    peft_config.inference_mode=True
                    peft_config.save_pretrained(os.path.join(args.output_dir,'{}step_ckpt'.format(completed_steps)))
                    peft_config.inference_mode=False
                    for file in os.listdir(args.output_dir):
                        if 'step_ckpt' in file and str(completed_steps) not in file and str(2200) not in file and '4400' not in file and '6600' not in file:
                            os.system('rm -rf '+ os.path.join(args.output_dir,file))
                
                
                
                # plan 1:
                # plan 2:
                # state_dict=accelerator.unwrap_model(model).state_dict()
                # accelerator.save(state_dict, os.path.join(args.output_dir,'{}step_model'.format(completed_steps)))
                # plan 3:
                # success = model.save_checkpoint(args.output_dir, "{}step_ckpt".format(completed_steps), {'epoch':epoch,'last_global_step':completed_steps})
                # status_msg = f"checkpointing: checkpoint_folder={args.output_dir}, ckpt_id={completed_steps}"
                # if success:
                #     logging.info(f"Success {status_msg}")
                # else:
                #     logging.warning(f"Failure {status_msg}")
        accelerator.wait_for_everyone()
        accelerator.save_state(output_dir = os.path.join(args.output_dir,'{}epoch_ckpt'.format(epoch)))
        
        peft_config.inference_mode=True
        peft_config.save_pretrained(os.path.join(args.output_dir,'{}epoch_ckpt'.format(epoch)))
        peft_config.inference_mode=False
            

if __name__ == "__main__":
    main()