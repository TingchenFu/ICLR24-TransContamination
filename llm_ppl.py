import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.optimization import (
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup
)
from torch.utils.data import DataLoader, Dataset
import argparse
import logging
from str2bool import str2bool
import random
import json
import datetime
import numpy as np
from tqdm import tqdm

from util import template_for_ppl,template_for_generation,  update_argument


template_part='\n[{}]: '
lang_dict={
    'en':'English',
    'zh':'Chinese',
    'eng_Latn':'English',
    'cat_Latn':'Catalan',
    'pan_Guru':'Eastern Panjabi',
    'ibo_Latn':'Igbo',
    'tsn_Latn':'Tswana',
    'zho_Hans':'Chinese'
}




class SimpleBatcher:
    def __init__(self, tokenizer,max_prompt_length,max_target_length,device,src_lang,tgt_lang) -> None:
        self.max_prompt_length = max_prompt_length
        self.max_target_length = max_target_length
        self.tokenizer=tokenizer
        self.device=device
        self.src_lang=src_lang
        self.tgt_lang=tgt_lang

    def promptppl(self,prompt_list,tgt_list,architecture):
        assert architecture in ['encoder-decoder','decoder-only','target-only']
        if architecture=='decoder-only':
            batch_input_id=[]
            batch_label=[]
            bs = len(prompt_list)
            for i in range(bs):
                src=self.tokenizer.encode(prompt_list[i],add_special_tokens=False, padding=False,truncation=True, max_length=self.max_prompt_length)
                tgt = self.tokenizer.encode(tgt_list[i], padding=False,truncation=True,max_length=self.max_target_length)
                input_id = src + tgt
                label = [-100] * len(src) + tgt
                batch_input_id.append(input_id)
                batch_label.append(label)
            
            max_length=max([len(x) for x in batch_input_id])
            for i in range(bs):
                padding_length = max_length - len(batch_input_id[i])
                if padding_length:
                    batch_input_id[i].extend([self.tokenizer.pad_token_id] * padding_length)
                    batch_label[i].extend([-100]*padding_length)
                assert len(batch_input_id[i]) == len(batch_label[i]) == max_length

            batch_input_id=torch.tensor(batch_input_id,dtype=torch.long,device=self.device)
            attention_mask=(batch_input_id!=self.tokenizer.pad_token_id).to(self.device)

            return{
                'input_id':batch_input_id,
                'attention_mask':attention_mask,
                'label':torch.tensor(batch_label,dtype=torch.long,device=self.device)
            }
        
        elif architecture=='target-only':
            batch_input_id=[]
            batch_label=[]
            bs = len(prompt_list)
            for i in range(bs):
                src=self.tokenizer.encode(prompt_list[i],add_special_tokens=True, padding=False,truncation=True, max_length=self.max_prompt_length)    
                tgt = self.tokenizer.encode(tgt_list[i], padding=False,truncation=True,max_length=self.max_target_length)
                if ' ' not in prompt_list[i] or ' ' not in tgt_list[i] or len(tgt)==1:
                    tgt=[self.tokenizer.pad_token_id]+tgt
                input_id=tgt
                label = tgt+[]
                batch_input_id.append(input_id)
                batch_label.append(label)
            
            max_length=max([len(x) for x in batch_input_id])
            for i in range(bs):
                padding_length = max_length - len(batch_input_id[i])
                if padding_length:
                    batch_input_id[i].extend([self.tokenizer.pad_token_id] * padding_length)
                    batch_label[i].extend([-100]*padding_length)
                assert len(batch_input_id[i]) == len(batch_label[i]) == max_length

            batch_input_id=torch.tensor(batch_input_id,dtype=torch.long,device=self.device)
            attention_mask=(batch_input_id!=self.tokenizer.pad_token_id).to(self.device)

            return{
                'input_id':batch_input_id,
                'attention_mask':attention_mask,
                'label':torch.tensor(batch_label,dtype=torch.long,device=self.device)
            }

        elif architecture=='encoder-decoder':
            tokenized_source=self.tokenizer.batch_encode_plus(prompt_list,max_length=self.max_prompt_length,truncation=True,padding='longest',return_tensors='pt')
            tokenized_target=self.tokenizer.batch_encode_plus(tgt_list,max_length=self.max_target_length,truncation=True,padding='longest',return_tensors='pt')
            label=[
                [(l if l!= self.tokenizer.pad_token_id else -100) for l in label] for label in tokenized_target['input_ids']
            ]
            return {
                'input_id':tokenized_source['input_ids'].to(self.device),
                'attention_mask':tokenized_source['attention_mask'].to(self.device),
                'label':torch.tensor(label,dtype=torch.long,device=self.device)
            }
        else:
            raise NotImplementedError


    
    def separate_promptppl(self,src_list,tgt_list,example_list,use_prompt,architecture):
        if architecture=='decoder-only':
            batch_input_id=[]
            batch_label=[]
            batch_language_mask=[]
            bs = len(src_list)
            if use_prompt:
                for i in range(bs):
                    src_id = []
                    language_mask = []
                    for example in example_list[i]: 
                        src_id += self.tokenizer.encode(template_part.format(lang_dict[self.src_lang]),add_special_tokens=False)
                        language_mask += [1]* (len(src_id) - len(language_mask))
                        src_id += self.tokenizer.encode(example['src'],add_special_tokens=False) 
                        language_mask += [1]* (len(src_id) - len(language_mask)) if (self.src_lang =='en' or self.src_lang == 'eng_Latn') else [0]* (len(src_id) - len(language_mask))
                        src_id += self.tokenizer.encode(template_part.format(lang_dict[self.tgt_lang]),add_special_tokens=False)
                        language_mask += [1]* (len(src_id) - len(language_mask)) 
                        src_id += self.tokenizer.encode(example['tgt'],add_special_tokens=False) 
                        language_mask += [1]* (len(src_id) - len(language_mask)) if (self.tgt_lang =='en' or self.tgt_lang == 'eng_Latn') else [0]* (len(src_id) - len(language_mask))

                    src_id += self.tokenizer.encode(template_part.format(lang_dict[self.src_lang]),add_special_tokens=False)
                    language_mask += [1]* (len(src_id) - len(language_mask))
                    src_id += self.tokenizer.encode(src_list[i],add_special_tokens=False) 
                    language_mask += [1]* (len(src_id) - len(language_mask)) if (self.src_lang =='en' or self.src_lang == 'eng_Latn') else [0]* (len(src_id) - len(language_mask))
                    src_id += self.tokenizer.encode(template_part.format(lang_dict[self.tgt_lang]),add_special_tokens=False)
                    language_mask += [1]* (len(src_id) - len(language_mask)) 

                    tgt_id = self.tokenizer.encode(tgt_list[i], padding=False, truncation=True, max_length=self.max_target_length) 
                    input_id = src_id[:self.max_prompt_length] + tgt_id[:self.max_target_length]
                    label= [-100] * len(src_id[:self.max_prompt_length]) + tgt_id[:self.max_target_length] 
                    language_mask = language_mask[:self.max_prompt_length]
                    language_mask += [1] * (len(input_id) - len(language_mask)) if (self.tgt_lang == 'en' or self.tgt_lang == 'eng_Latn') else [0] * (len(input_id) - len(language_mask))

                    assert len(language_mask) == len(input_id) == len(label)
                    batch_input_id.append(input_id)
                    batch_label.append(label)
                    batch_language_mask.append(language_mask)

            else:# not use prompt
                for i in range(bs):
                    src_id = self.tokenizer.encode(src_list[i],add_special_tokens=False)
                    language_mask = [1] * len(src_id) if (self.src_lang == 'en' or self.src_lang == 'eng_Latn') else [0] * len(src_id)
                    tgt_id = self.tokenizer.encode(tgt_list[i])
                    input_id = src_id[:self.max_prompt_length] + tgt_id[:self.max_target_length]
                    language_mask = language_mask[:self.max_prompt_length]
                    language_mask += [1] * (len(input_id) - len(language_mask)) if (self.tgt_lang =='en' or self.tgt_lang == 'eng_Latn') else  [0] * (len(input_id) - len(language_mask))
                    label = [-100] * len(src_id[:self.max_prompt_length]) +  tgt_id[:self.max_target_length]

                    assert len(language_mask) == len(input_id) == len(label)
                    batch_input_id.append(input_id)
                    batch_label.append(label)
                    batch_language_mask.append(language_mask)

            
            max_length=max([len(x) for x in batch_input_id])
            for i in range(bs):
                padding_length = max_length - len(batch_input_id[i])
                if padding_length:
                    batch_input_id[i].extend([self.tokenizer.pad_token_id] * padding_length)
                    batch_label[i].extend([-100]*padding_length)
                    batch_language_mask[i].extend([0]*padding_length)
                assert len(batch_input_id[i]) == len(batch_label[i]) == len(batch_language_mask[i]) == max_length 

            batch_input_id=torch.tensor(batch_input_id,dtype=torch.long,device=self.device)
            attention_mask=(batch_input_id!=self.tokenizer.pad_token_id).to(self.device)
            batch_language_mask = torch.tensor(batch_language_mask,dtype=torch.long,device = self.device)

            return{
                'input_id':batch_input_id,
                'attention_mask':attention_mask,
                'label':torch.tensor(batch_label,dtype=torch.long,device=self.device),
                'language_mask':batch_language_mask
            }
        
        
        elif architecture=='target-only':
            # only the target transltion, no src, no prompt
            batch_input_id=[]
            batch_label=[]
            batch_language_mask=[]
            bs = len(src_list)
            for i in range(bs):
                tgt = self.tokenizer.encode(tgt_list[i], padding=False,truncation=True,max_length=self.max_target_length)
                if ' ' not in src_list[i] or ' ' not in tgt_list[i] or len(tgt)==1:
                    tgt=[self.tokenizer.pad_token_id]+tgt
                input_id=tgt
                label = tgt+[]
                batch_input_id.append(input_id)
                batch_label.append(label)
                if self.tgt_lang == 'en' or self.tgt_lang == 'eng_Latn':
                    batch_language_mask.append([1]*len(input_id))
                else:
                    batch_language_mask.append([0]*len(input_id))
                    
            
            max_length=max([len(x) for x in batch_input_id])
            for i in range(bs):
                padding_length = max_length - len(batch_input_id[i])
                if padding_length:
                    batch_input_id[i].extend([self.tokenizer.pad_token_id] * padding_length)
                    batch_label[i].extend([-100]*padding_length)
                    batch_language_mask[i].extend([0]*padding_length)
                assert len(batch_input_id[i]) == len(batch_label[i]) == len(batch_language_mask[i]) == max_length 

            
            batch_input_id=torch.tensor(batch_input_id,dtype=torch.long,device=self.device)
            attention_mask=(batch_input_id!=self.tokenizer.pad_token_id).to(self.device)
            batch_language_mask = torch.tensor(batch_language_mask,dtype=torch.long,device = self.device)

            return{
                'input_id':batch_input_id,
                'attention_mask':attention_mask,
                'label':torch.tensor(batch_label,dtype=torch.long,device=self.device),
                'language_mask':batch_language_mask
            }

    
    def language_modeling(self,text_list,mask):
        # no: no loss mask
        # half: mask the right-half loss
        # rand: mask random length loss
        assert mask in ['no','half','rand']
        bs=len(text_list)
        batch_input_id=[]
        batch_label=[]
        for i in range(bs):
            input_id = self.tokenizer.encode(text_list[i],max_length=self.max_prompt_length+self.max_target_length,truncation=True,padding=False)
            if mask=='no':
                label=input_id+[]
            elif mask=='rand':
                mask_idx = random.choices(range(len(input_id)),k= len(input_id)//2)
                label=[input_id[x] if x not in mask_idx else -100 for x in range(len(input_id))]
            elif mask=='half':
                label=input_id[:(len(input_id))//2]
                label+= [-100]* (len(input_id)-len(label))
            assert len(label)==len(input_id)

            batch_input_id.append(input_id)
            batch_label.append(label)
        
        max_length=max([len(x) for x in batch_input_id])
        for i in range(bs):
            padding_length = max_length - len(batch_input_id[i])
            if padding_length:
                batch_input_id[i].extend([self.tokenizer.pad_token_id] * padding_length)
                batch_label[i].extend([-100]*padding_length)
            assert len(batch_input_id[i]) == len(batch_label[i]) == max_length

        batch_input_id=torch.tensor(batch_input_id,dtype=torch.long,device=self.device)
        attention_mask=(batch_input_id!=self.tokenizer.pad_token_id).to(self.device)

        return{
            'input_id':batch_input_id,
            'attention_mask':attention_mask,
            'label':torch.tensor(batch_label,dtype=torch.long,device=self.device)
        }




logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Pre-training for Knowledge-Grounded Conversation')
parser.add_argument("--debug",default=False,type=str2bool,help='debug mode, using small dataset')

parser.add_argument("--dataset",type=str,choices=['WMTnews21','flores200','MUSE'],default='WMTnews21')
parser.add_argument("--source_language",type=str,default='zh')
parser.add_argument("--target_language",type=str,default='en')


# training scheme
parser.add_argument('--train_batch_size', type=int, default=4)
parser.add_argument('--eval_batch_size', type=int, default=1)
parser.add_argument('--n_step', type=int, default=1000000)
parser.add_argument('--accum_step', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--clip', type=float, default=2.0)
parser.add_argument('--schedule', type=str, default='cosine')
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--adam_epsilon', type=float, default=1e-8)
parser.add_argument('--warmup_step', type=int, default=500)
parser.add_argument('--n_epoch', type=int, default=3)

# print and logging
parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--eval_every', type=int, default=2500)

# input
parser.add_argument("--max_input_length",type=int,default=1024-128)
parser.add_argument("--max_target_length",type=int,default=128)
parser.add_argument("--example_fn",type=str,default='rand')
parser.add_argument("--n_example",type=int,default=0)
parser.add_argument("--use_prompt",type=str2bool,default=True)
parser.add_argument("--disturb",type=str2bool,default=False)
parser.add_argument("--architecture",type=str,default='target-only',choices=['encoder-decoder','decoder-only','target-only'])


# model
# parser.add_argument("--model",type=str,choices=['mBART-large-cc25','mT5-base','mGPT','bloom-560m','bloomz-560m','bloom-7b','bloom-3b','bloom-1b1','bloom-1b7','bloom-175b'],default='bloom-560m')

parser.add_argument("--model_name_or_path",type=str,default=None)
parser.add_argument("--ckpt_path",type=str,default=None)
parser.add_argument("--peft_model_path",type=str,default=None)


args = parser.parse_args()
args = update_argument(args,'promptppl')


if args.debug:
    args.print_every=1
    args.eval_every=8
    args.train_batch_size=2
    args.eval_batch_size=2


print("\nParameters:")
for attr, value in sorted(vars(args).items()):
    print("{}={}".format(attr.upper(), value))


torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark=True
from dataset import MTPromptDataset

# Build dataset
time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("Create dataset begin... | %s " % time_str)

if args.n_example <=1:
    args.eval_batch_size =1
else:
    args.eval_batch_size =1 

dev_dataset = MTPromptDataset(args.dev_src_file,args.dev_tgt_file,args.debug,False)
test_dataset= MTPromptDataset(args.test_src_file,args.test_tgt_file,args.debug,args.disturb)
test_loader = DataLoader(test_dataset,batch_size=args.eval_batch_size,shuffle=True,collate_fn=MTPromptDataset.collate_fn)

time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("Create dataset end... | %s " % time_str)

print('Eval Dataset {}'.format(len(test_dataset)))
print(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
# tokenizer.padding_side = 'right'
batcher = SimpleBatcher(tokenizer,args.max_input_length,args.max_target_length,device,args.source_language,args.target_language)


if args.ckpt_path:
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    if not args.ckpt_path.endswith('bin'):
        args.ckpt_path = os.path.join(args.ckpt_path,'pytorch_model.bin')
    reloaded=torch.load(args.ckpt_path)
    try:
        reloaded["lm_head.weight"] = reloaded["transformer.word_embeddings.weight"]
    except:
        reloaded["zh_lm_head.weight"] = reloaded["transformer.zh_word_embeddings.weight"]
        reloaded["en_lm_head.weight"] = reloaded["transformer.en_word_embeddings.weight"]
    model.load_state_dict(reloaded,strict=True)
    print("ckpt model loaded")

elif args.peft_model_path:
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,device_map='auto',torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model,args.peft_model_path,device_map="auto",torch_dtype=torch.float16)
    print("peft model loaded")

else:
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, 
                                                #device_map='auto', 
                                                #torch_dtype=torch.float16,
                                                #offload_folder='./offload',
                                                )

# configuration = CONFIG2FN[args.model].from_pretrained(args.model_path)
# with init_empty_weights():
#     model = MODEL2FN(configuration)
# if args.ckpt_path:
#     model = load_checkpoint_and_dispatch(model, args.ckpt_path, device_map="auto", offload_folder='offload',offload_state_dict=True)
# else:
#     model = load_checkpoint_and_dispatch(model, args.model_path, device_map="auto", offload_folder='offload',offload_state_dict=True)



# model = MODEL2FN[args.model].from_pretrained(args.model_path,device_map='auto')
# reloaded=torch.load('/apdcephfs/share_916081/shared_info/tingchenfu/work1/dump/scratch_puremix_bloom-560m/seq1024bs512lr0.0003warm500cosine_seperate6/8627step_ckpt/pytorch_model.bin')
# reloaded["lm_head.weight"] = reloaded["transformer.word_embeddings.weight"]
# model.load_state_dict(reloaded,strict=True)
#model.to(device)


if args.ckpt_path:
    ppl_output = args.ckpt_path.replace('ckpt/pytorch_model.bin','')+args.dataset+'_'+args.source_language+args.target_language+'_'+str(args.n_example)+'shot_ppl.json'
elif args.peft_model_path:
    ppl_output = args.peft_model_path.replace('ckpt','')+args.dataset+'_'+args.source_language+args.target_language +'_' +str(args.n_example)+'shot_ppl.json'
else:
    ppl_output = os.path.join('/apdcephfs/share_916081/shared_info/tingchenfu/work1/dump/{}'.format(args.model_name_or_path.split('/')[-1]),args.dataset+'_'+args.source_language+args.target_language +'_' +str(args.n_example)+'shot_ppl.json')

print("ppl output path:  {}".format(ppl_output))

torch.cuda.empty_cache()

def prompt_step():
    #sentence_ppl = 0.0
    #f=open('/data/home/tingchenfu/work1/data/wmt21{}{}{}example.jsonl'.format(args.source_language,args.target_language, args.n_example),'w',encoding='utf-8')
    word_loss = 0.0
    all_word = 0
    model.eval()
    model.cuda()
    sent_ppl=[]
    with torch.no_grad():
        for src_list,tgt_list in tqdm(test_loader):
            prompt_list=[]
            bs=len(src_list)
            for i in range(bs):
                examples=dev_dataset.get_example(src_list[i],tgt_list[i],args.example_fn,args.n_example,args.seed)
                prompt=template_for_ppl(examples,src_list[i],args.source_language,args.target_language)
                #f.write(json.dumps({'prompt':prompt,'completion':tgt_list[i]},ensure_ascii=False)+'\n')
                prompt_list.append(prompt)
            if args.use_prompt:
                batch_dict=batcher.promptppl(prompt_list,tgt_list,args.architecture)
            else:
                batch_dict=batcher.promptppl(src_list,tgt_list,args.architecture)

            loss=model(input_ids=batch_dict['input_id'].cuda(),attention_mask=batch_dict['attention_mask'].cuda(),labels=batch_dict['label'].cuda(),return_dict=True)['loss']
            # if torch.isnan(loss):
            #     print(batch_dict['input_id'])
            #     print(batch_dict['label'])
            n_word = (batch_dict['label']!=-100).sum(1).sum(0)
            word_loss += (loss*n_word).to(torch.float32)
            
            sent_ppl.append(torch.exp(loss).item())
            all_word += n_word

    word_ppl = torch.exp(word_loss/all_word).item()
    print("word ppl {}".format(word_ppl))
    json.dump(sent_ppl,open(ppl_output,'w'))

prompt_step()
#f.close()

print("ppl output path:  {}".format(ppl_output))