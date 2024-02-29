import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformers import MBartTokenizer, MBartConfig, MBartForConditionalGeneration
# from transformers import MT5Tokenizer, MT5Config, MT5ForConditionalGeneration
# from transformers import GPT2LMHeadModel,GPT2Config
# from transformers import BloomConfig,BloomForCausalLM,BloomTokenizerFast
# from transformers import LlamaConfig,LlamaForCausalLM,LlamaTokenizer
# from transformers import XGLMConfig,XGLMForCausalLM,XGLMTokenizerFast
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation import GenerationConfig
import os
from peft import PeftModel, PeftConfig
from torch.utils.data import DataLoader, Dataset
import argparse
import logging
from str2bool import str2bool
import random 
import json
import datetime
import numpy as np
from tqdm import tqdm

from util import template_for_generation,update_argument

from accelerate import load_checkpoint_and_dispatch
from accelerate import init_empty_weights
from accelerate import infer_auto_device_map


# TOKENIZER2FN={
#     'mBART-large-cc25':MBartTokenizer,
#     'mT5-base':MT5Tokenizer,
#     'mGPT':MT5Tokenizer,
#     'bloom-560m':BloomTokenizerFast,
#     'bloomz-560m':BloomTokenizerFast,
#     'bloom-7b':BloomTokenizerFast,
#     'bloomz-7b':BloomTokenizerFast,
#     'llama-7b':LlamaTokenizer,
#     'xglm-7b':XGLMTokenizerFast,
#     'bloom-175b':BloomTokenizerFast,
# }

# MODEL2FN={
#     'mBART-large-cc25':MBartForConditionalGeneration,
#     'mT5-base':MT5ForConditionalGeneration,
#     'mGPT':GPT2LMHeadModel,
#     'bloom-560m':BloomForCausalLM,
#     'bloomz-560m':BloomForCausalLM,
#     'bloom-7b':BloomForCausalLM,
#     'bloomz-7b':BloomForCausalLM,
#     'llama-7b':LlamaForCausalLM,
#     'xglm-7b':XGLMForCausalLM,
#     'bloom-175b':BloomForCausalLM,
# }

# CONFIG2FN={
#     'bloom-560m':BloomConfig,
#     'bloomz-560m':BloomConfig,
#     'bloom-7b':BloomConfig,
#     'bloomz-7b':BloomConfig,
#     'llama-7b':LlamaConfig,
#     'xglm-7b':XGLMConfig,
#     'bloom-175b':BloomConfig,
# }

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Pre-training for Knowledge-Grounded Conversation')
parser.add_argument("--debug",default=False,type=str2bool,help='debug mode, using small dataset')

parser.add_argument("--dataset",type=str,choices=['WMTnews21','flores200'],default='flores200')
parser.add_argument("--source_language",type=str,choices=['zh','en','eng_Latn','cat_Latn','pan_Guru','ibo_Latn','tsn_Latn','zho_Hans'],default='zh')
parser.add_argument("--target_language",type=str,choices=['zh','en','eng_Latn','cat_Latn','pan_Guru','ibo_Latn','tsn_Latn','zho_Hans'],default='en')

# input
parser.add_argument('--seed',type=int,default=0)
parser.add_argument('--eval_batch_size', type=int, default=1)
parser.add_argument("--max_input_length",type=int,default=1024)
parser.add_argument("--max_target_length",type=int,default=128)
parser.add_argument("--example_fn",type=str,default='rand')
parser.add_argument("--n_example",type=int,default=0)
parser.add_argument("--architecture",type=str,default='target-only',choices=['encoder-decoder','decoder-only','target-only'])

parser.add_argument("--model_name_or_path",type=str,default='/apdcephfs/share_916081/shared_info/tingchenfu/PLM/bloom-560m')
parser.add_argument("--peft_model_path",type=str,default=None)
parser.add_argument("--ckpt_path",type=str,default=None)
parser.add_argument("--revision_name",type=str,default=None)
# parser.add_argument("--config_path",type=str,default=None)
# parser.add_argument("--vocab_path",type=str,default=None)

args = parser.parse_args()
args = update_argument(args,'promptppl')

if args.debug:
    args.print_every=1
    args.eval_every=8
    args.train_batch_size=2
    args.eval_batch_size=2

# os.makedirs(args.output_dir,exist_ok=True)
# logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "log"), 'w'))


print("\nParameters:")
for attr, value in sorted(vars(args).items()):
    print("{}={}".format(attr.upper(), value))


torch.cuda.empty_cache()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark=True
from dataset import MTPromptDataset


dev_dataset = MTPromptDataset(args.dev_src_file,args.dev_tgt_file,args.debug)
test_dataset= MTPromptDataset(args.test_src_file, args.test_tgt_file, args.debug)
test_loader = DataLoader(test_dataset,batch_size=args.eval_batch_size,shuffle=False,collate_fn=MTPromptDataset.collate_fn)


tokenizer = AutoTokenizer.from_pretrained('/apdcephfs/share_916081/shared_info/tingchenfu/PLM/bloom-560m')
tokenizer.pad_token_id = (0)
tokenizer.padding_side = 'right'



free_gb=int(torch.cuda.mem_get_info()[0]/1024**3)
max_memory = f'{free_gb-3}GB'
n_gpus = torch.cuda.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}
max_memory['cpu']='30GiB'

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16
# )
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                            device_map = 'auto',
                                            torch_dtype = torch.float16,
                                            revision=args.revision_name,
                                            local_files_only=True,
                                            #load_in_8bit = True,
                                            #quantization_config=quantization_config
                                            #offload_folder="./offload", 
                                            # offload_state_dict = True,
                                            #max_memory = max_memory
                                            )

# model.to_bettertransformer()
# print("model to better transformer")

# configuration = BloomConfig.from_json_file(os.path.join(args.model_path,'config.json'))
# with init_empty_weights():
#     model = BloomForCausalLM(configuration)
# model.tie_weights() 
# max_memory={0: "30GiB", 1: "30GiB", 2: "30GiB", 3: "30GiB", 4: "30GiB", 5: "30GiB", 6: "30GiB", 7: "30GiB", "cpu": "150GiB"}
# device_map = infer_auto_device_map(model, no_split_module_classes=["BloomBlock"],max_memory=max_memory)
# model = load_checkpoint_and_dispatch(model, args.model_path, device_map=device_map, dtype=torch.float16,offload_folder='./offload')




if args.peft_model_path:
    model = PeftModel.from_pretrained(model,args.peft_model_path,device_map="balanced_low_0",torch_dtype=torch.float16)
    print("peft model loaded")
elif args.ckpt_path:
    if not args.ckpt_path.endswith('bin'):
        args.ckpt_path = os.path.join(args.ckpt_path, 'pytorch_model.bin')
    reloaded=torch.load(args.ckpt_path)
    reloaded["lm_head.weight"] = reloaded["transformer.word_embeddings.weight"]
    model.load_state_dict(reloaded,strict=True)
    print("checkpoint loaded")
else:
    print("original model")



if args.ckpt_path:
    decode_output = args.ckpt_path.replace('ckpt/pytorch_model.bin','')+args.dataset+'_'+args.source_language+args.target_language+'_'+str(args.n_example)+'shot_raw.json'
elif args.peft_model_path:
    decode_output = args.peft_model_path.replace('ckpt','')+args.dataset+'_'+args.source_language+args.target_language +'_' +str(args.n_example)+'shot_raw.json'
elif args.revision_name:
    decode_output = os.path.join('/apdcephfs/share_916081/shared_info/tingchenfu/work1/dump/{}'.format(args.model_name_or_path.split('/')[-1]+'_'+args.revision_name),args.source_language+args.target_language+str(args.n_example)+'shot_raw.json')
else:
    decode_output = os.path.join('/apdcephfs/share_916081/shared_info/tingchenfu/work1/dump/{}'.format(args.model_name_or_path.split('/')[-1]),args.dataset+'_'+args.source_language+args.target_language+str(args.n_example)+'shot_raw.json')

print(f"decode output: {decode_output}")
os.makedirs(   '/'.join(decode_output.split('/')[:-1]), exist_ok=True)


def generate_step():
    hypothesis=[]
    model.eval()
    with torch.no_grad():
        for src_list,tgt_list in tqdm(test_loader):
            prompt_list=[]
            bs=len(src_list)
            for i in range(bs):
                examples=dev_dataset.get_example(src_list[i],tgt_list[i],args.example_fn,args.n_example,args.seed)
                prompt=template_for_generation(examples,src_list[i],args.source_language,args.target_language)
                #f.write(json.dumps({'prompt':prompt,'completion':tgt_list[i]},ensure_ascii=False)+'\n')
                prompt_list.append(prompt)

            tokenized = tokenizer.batch_encode_plus(prompt_list,max_length=args.max_input_length,truncation=True,padding='longest',return_tensors='pt')
            input_ids = tokenized['input_ids'].to(model.device)

            input_length = input_ids.shape[1]    
            # loss=model(input_ids=input_ids,attention_mask=attention_mask,labels=input_ids).loss
            # print(loss)
            assert input_length > 0
            output = model.generate(input_ids=input_ids,max_new_tokens = 64, min_new_tokens=5)
            hyp = tokenizer.batch_decode(output[:,input_length:].detach().cpu().tolist(), skip_special_tokens=True,clean_up_tokenization_spaces=False)
            hypothesis.extend(hyp)
            #print('*******************************************************')
            decode_id = input_ids.detach().cpu().tolist()[0]
            # print(decode_id)
            #print(hyp[0])
    return hypothesis

hypothesis=generate_step()


with open(decode_output,'w') as f:
    json.dump(hypothesis,f)
print(f"decode output: {decode_output}")