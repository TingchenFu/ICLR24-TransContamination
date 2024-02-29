# coding=utf-8
import sys
# sys.path.append('/data/home/tingchenfu/work1')
# sys.path.append('/data/home/tingchenfu/work1/code')
sys.path.append('/apdcephfs/share_916081/tingchenfu/work1/data')
import os
import json
from tqdm import tqdm
# import langdetect
# import ftlangdetect
import re
import evaluate
from translate import translate
import multiprocessing
from multiprocessing import Pool




# from transformers import GPT2Config,GPT2LMHeadModel
# config=GPT2Config.from_json_file('/apdcephfs/share_916081/tingchenfu/PLM/mGPT/config.json')
# model=GPT2LMHeadModel(config)
# reloaded=torch.load('/apdcephfs/share_916081/tingchenfu/work1/dump/debug/16step_model')
# model.load_state_dict(reloaded,strict=True)

# from transformers import MT5Tokenizer
# tokenizer=MT5Tokenizer.from_pretrained('/apdcephfs/share_916081/tingchenfu/PLM/mT5-base')

# encoded=tokenizer.encode('I love coding and training large language model')
# print(encoded)

pattern=re.compile(r'[A-Za-z0-9 _\,\.\;\:\'\"\/\?\!\-#\&]*[a-zA-Z]{3,}[A-Za-z0-9 _\,\.\;\:\'\"\/\?\!\-#\&]*')

english_dict={}
fdict=open('/apdcephfs/share_916081/shared_info/tingchenfu/work1/data/ldc_ec_dict.2.0.txt')
for line in fdict.readlines():
    raw,translated=line.strip('\n').split('\t')[:2]
    translated = translated.strip('/').split('/')[0]
    if raw.upper() == raw:
        english_dict[raw]=translated
    else:
        english_dict[raw.lower()]=translated
fdict.close()


# import thulac
# thu = thulac.thulac(seg_only=True)
import jieba


# print('here is ok')
# exit()


def check_cinese(string):
    '''
    detect chinese fragment from a single line of text
    '''
    results=[]
    start=None
    end=None
    for i in range(len(string)):
        if u'\u4e00' <=string[i] <= u'\u9fa5':
            if start== None:
                start=i
        elif string[i] not in ['！','？','，','。','、','·','……','‘','’','“','”','%','（','）',] and not string[i].isdigit():
            if start!=None:
                end=i
                try:
                    #if end-start>5 and langdetect.detect(string[start:end])=='zh-cn' and  ftlangdetect.detect(string[start:end])['lang']=='zh':
                    results.append(string[start:end])
                except:
                    pass
                start=None
                end=None
    if start is not None and len(string) - start > 5:
        results.append(string[start:len(string)])
    return results

def check_english(string):
    '''
    detect english fragment from a single line of text
    '''
    replace_list=[',','.','?','!','$','(',')','*','~','@','%','-','+','=','\\','/']
    results=[]
    matched=pattern.findall(string)
    for m in matched:
        if len(m) < 20:
            continue
        pure=m
        for rep in replace_list:
            pure=pure.replace(rep,' ')
        if not any([word.lower() in english_dict.keys() or word.upper() in english_dict.keys()    for word in pure.split(' ')]):
            continue
        if 'http' in m or '.com' in m or 'www' in m or '.cn' in m or '.net' in m:
            continue
        if m.upper()==m:
            continue
        if 'srk:' in m or 'src=' in m or 'img=' in m:
            continue
        if 'all rights reserved' in m.lower():
            continue
        try:
            #if langdetect.detect(m)=='en' and  ftlangdetect.detect(m)['lang']=='en':
            results.append(m)
        except:
            pass

    return results


def sim(hypothesis,references):
    '''
    hypothesis: a str
    references: a list of str
    '''
    try:
        sacrebleu = evaluate.load("sacrebleu")
        results = sacrebleu.compute(predictions=[hypothesis], references=[references])
        return round(results["score"], 1)
    except:
        print(hypothesis)
        print(references)
        return 0.0





def parsing_from_english(inputs):
    '''
    parsing chinese from english
    '''
    doc,doc_no = inputs
    # fin=open(input_file,'r',encoding='utf-8')
    # fout_sent=open(sent_output,'w',encoding='utf-8')
    # fout_word=open(word_output,'w',encoding='utf-8')
    # fout_switch=open(swith_output,'w',encoding='utf-8')
    # count_sent_align=0
    # count_word_align=0
    # f=open('/data/home/tingchenfu/work1/data/mC4/c4/multilingual/c4-en.tfrecord-00001-of-11264.json')
    # #f=open('./text')
    # fout1=open('./temp_sent','w',encoding='utf-8')
    # fout2=open('./temp_word','w',encoding='utf-8')
    # english_dict={}
    word_align=[]
    sent_align=[]
    code_switch=[]

    # doc_no=-1
    #for record in tqdm(fin.readlines()):
    # doc_no+=1
    try:
        text = json.loads(doc)['text']
    except:
        text=doc
    lines=text.split('\n')
    
    flag=[0]*len(lines)
    sent_align_flag=[0]*len(lines)
    word_align_flag=[0]*len(lines)


    english_cache = dict()
    for i in range(len(lines)):
        fragments = check_chinese(lines[i])
        if len(fragments)==0:
            continue
        if i-1 >=0: flag[i-1]=1
        flag[i]=1
        if i+1 < len(lines) : flag[i+1]=1
        candidates=[]
        for j in range(max(i-1,0),min(i+1,len(lines))):
            if j not in english_cache.keys():
                english_cache[j]=check_english(lines[j])
            candidates.extend(english_cache[j])
        if len(candidates)==0:
            continue
        
        translations = translate(fragments, srcLang="zh", tgtLang="en")
        for frag, tran in zip(fragments,translations):
            if len(frag) > 5:
                # judge sentence align    
                if sim(tran,candidates) > 10:
                    if i-1 >=0: sent_align_flag[i-1]=1
                    sent_align_flag[i]=1
                    if i+1 < len(lines): sent_align_flag[i+1]=1
                    break

            # judge word align
            if any([word in ' '.join(candidates).split(' ') for word in tran.split(' ') ]):
                if i-1 >=0: word_align_flag[i-1]=1
                word_align_flag[i]=1
                if i+1 < len(lines): word_align_flag[i+1]=1
                break

    start=None
    end=None
    for i in range(len(lines)):
        if start is None and sent_align_flag[i]:
            start=i
        elif start is not None and not sent_align_flag[i]:
            end=i
            sent_align.append('\n'.join(lines[start:end]))
            #fout_sent.write(json.dumps({'file_no':input_file.split('/')[-1],'doc_no': doc_no,'text':'\n'.join(lines[start:end])},ensure_ascii=False)+'\n')
            start=None
    if start is not None:
        sent_align.append('\n'.join(lines[start:len(lines)]))
        #fout_sent.write(json.dumps({'file_no':input_file.split('/')[-1],'doc_no': doc_no,'text':'\n'.join(lines[start:len(lines)])},ensure_ascii=False)+'\n')
    
    start=None
    end=None    
    for i in range(len(lines)):
        if start is None and word_align_flag[i]:
            start=i
        elif start is not None and not word_align_flag[i]:
            end=i
            word_align.append('\n'.join(lines[start:end]))
            #fout_word.write(json.dumps({'file_no':input_file.split('/')[-1],'doc_no': doc_no,'text':'\n'.join(lines[start:end])},ensure_ascii=False)+'\n')
            start=None
    if start is not None:
        word_align.append('\n'.join(lines[start:len(lines)]))
        #fout_word.write(json.dumps({'file_no':input_file.split('/')[-1],'doc_no': doc_no,'text':'\n'.join(lines[start:len(lines)])},ensure_ascii=False)+'\n')

    
    start=None
    end=None    
    for i in range(len(lines)):
        if start is None and flag[i] and not word_align_flag[i] and not sent_align_flag[i]:
            start=i
        elif start is not None and ((not flag[i]) or sent_align_flag[i] or word_align_flag[i]):
            end=i
            code_switch.append('\n'.join(lines[start:end]))
            #code_switch.write(json.dumps({'file_no':input_file.split('/')[-1],'doc_no': doc_no,'text':'\n'.join(lines[start:end])},ensure_ascii=False)+'\n')
            start=None
    if start is not None:
        code_switch.append('\n'.join(lines[start:len(lines)]))
        #fout_switch.write(json.dumps({'file_no':input_file.split('/')[-1],'doc_no': doc_no,'text':'\n'.join(lines[start:len(lines)])},ensure_ascii=False)+'\n')

    return sent_align,word_align,code_switch,doc_no



def parsing_from_chinese(inputs):
    doc,doc_no = inputs
    # input_file,sent_output,word_output,swith_output):
    # fin=open(input_file,'r',encoding='utf-8')
    # fout_sent=open(sent_output,'w',encoding='utf-8')
    # fout_word=open(word_output,'w',encoding='utf-8')
    # fout_switch=open(swith_output,'w',encoding='utf-8')
    # count_sent_align=0
    # count_word_align=0



    # f=open('/data/home/tingchenfu/work1/data/mC4/c4/multilingual/c4-en.tfrecord-00001-of-11264.json')
    # #f=open('./text')
    # fout1=open('./temp_sent','w',encoding='utf-8')
    # fout2=open('./temp_word','w',encoding='utf-8')
    # english_dict={}
    # fdict=open('/data/home/tingchenfu/work1/data/ldc_ec_dict.2.0.txt')
    # for line in fdict.readlines():
    #     raw,translated=line.strip('\n').split('\t')
    #     translated=translated.split('/')[1]
    #     if raw.upper() == raw:
    #         english_dict[raw]=translated
    #     else:
    #         english_dict[raw.lower()]==translated
    # fdict.close()

    word_align=[]
    sent_align=[]
    code_switch=[]
    
    try:
        text = json.loads(doc)['text']
    except:
        text= doc
    lines=text.split('\n')
    
    flag=[0]*len(lines)
    sent_align_flag=[0]*len(lines)
    word_align_flag=[0]*len(lines)


    chinese_cache = dict()
    for i in range(len(lines)):
        fragments = check_english(lines[i])
        if len(fragments)==0:
            continue
        candidates=[]
        for j in range(max(i-1,0),min(i+1,len(lines))):
            if j not in chinese_cache.keys():
                chinese_cache[j]=check_chinese(lines[j])
            candidates.extend(chinese_cache[j])
        if len(candidates)==0:
            continue
        if i-1 >=0: flag[i-1]=1
        flag[i]=1
        if i+1 < len(lines) : flag[i+1]=1
        translations = translate(fragments, srcLang="en", tgtLang="zh")
        for frag, tran in zip(fragments,translations):
            if len(frag.split(' ')) > 5:
                # judge sentence align    
                if sim(tran,candidates) > 10:
                    if i-1 >=0: sent_align_flag[i-1]=1
                    sent_align_flag[i]=1
                    if i+1 < len(lines): sent_align_flag[i+1]=1
                    break

            # judge word align
            #cutted_tran = [x for x in thu.cut(tran, text=True).split(' ') if len(x)>=2 and not x.isdigit() and all([u'\u4e00' <= xx <= u'\u9fa5'] for xx in x )    ]
            cutted_tran = [x for x in  list(jieba.cut(tran))    if len(x)>=2 and not x.isdigit() and all([u'\u4e00' <= xx <= u'\u9fa5'] for xx in x ) ]
            if any([ character in ''.join(candidates) for character in cutted_tran ]):
                if i-1 >=0: word_align_flag[i-1]=1
                word_align_flag[i]=1
                if i+1 < len(lines): word_align_flag[i+1]=1
                break

    start=None
    end=None
    for i in range(len(lines)):
        if start is None and sent_align_flag[i]:
            start=i
        elif start is not None and not sent_align_flag[i]:
            end=i
            sent_align.append('\n'.join(lines[start:end]))
            #sent_align.write(json.dumps({'file_no':input_file.split('/')[-1],'doc_no': doc_no,'text':'\n'.join(lines[start:end])},ensure_ascii=False)+'\n')
            start=None
    if start is not None:
        sent_align.append('\n'.join(lines[start:len(lines)]))
        #sent_align.write(json.dumps({'file_no':input_file.split('/')[-1],'doc_no': doc_no,'text':'\n'.join(lines[start:len(lines)])},ensure_ascii=False)+'\n')
    
    start=None
    end=None    
    for i in range(len(lines)):
        if start is None and word_align_flag[i]:
            start=i
        elif start is not None and not word_align_flag[i]:
            end=i
            word_align.append('\n'.join(lines[start:end]))
            #fout_word.write(json.dumps({'file_no':input_file.split('/')[-1],'doc_no': doc_no,'text':'\n'.join(lines[start:end])},ensure_ascii=False)+'\n')
            start=None
    if start is not None:
        word_align.append('\n'.join(lines[start:len(lines)]))
        #fout_word.write(json.dumps({'file_no':input_file.split('/')[-1],'doc_no': doc_no,'text':'\n'.join(lines[start:len(lines)])},ensure_ascii=False)+'\n')

    start=None
    end=None    
    for i in range(len(lines)):
        if start is None and flag[i] and not word_align_flag[i] and not sent_align_flag[i]:
            start=i
        elif start is not None and ((not flag[i]) or word_align_flag[i] or sent_align_flag[i]):
            end=i
            code_switch.append('\n'.join(lines[start:end]))
            #fout_switch.write(json.dumps({'file_no':input_file.split('/')[-1],'doc_no': doc_no,'text':'\n'.join(lines[start:end])},ensure_ascii=False)+'\n')
            start=None
    if start is not None:
        code_switch.append('\n'.join(lines[start:len(lines)]))
        #fout_switch.write(json.dumps({'file_no':input_file.split('/')[-1],'doc_no': doc_no,'text':'\n'.join(lines[start:len(lines)])},ensure_ascii=False)+'\n')
    
    return sent_align,word_align,code_switch,doc_no



#parsing_from_english('/data/home/tingchenfu/work1/data/mC4/c4/multilingual/c4-en.tfrecord-00001-of-11264.json','./temp_sent','./temp_word','./temp_switch')
#parsing_from_chinese('/data/home/tingchenfu/work1/data/mC4/c4/multilingual/c4-zh.tfrecord-00000-of-01024.json','./temp_sent_debug','./temp_word_debug','./temp_switch_debug')


if __name__ =='__main__':

    try:
        input_file=sys.argv[1]
        sent_output_file=sys.argv[2]
        word_output_file=sys.argv[3]
        switch_output_file=sys.argv[4]
        parsing_from = sys.argv[5]
    
    except:
        input_file='/data/home/tingchenfu/work1/data/mC4/c4/multilingual/c4-en.tfrecord-00001-of-11264.json'
        sent_output_file='./temp_sent'
        word_output_file='./temp_word'
        switch_output_file='./temp_switch'
        parsing_from = 'en'
    
    cores = multiprocessing.cpu_count()
    print("Using {} cores to run on instances".format(cores,))

# def nofun(no):
#     return '0'*(5-len(str(no)))+str(no)
# input_files= [os.path.join('/apdcephfs/share_916081/tingchenfu/Dataset/mC4/English','c4-en.tfrecord-{}-of-11264.json'.format(nofun(x))) for x in range(8)]
# sent_outputs = [os.path.join('/data/home/tingchenfu/work1/data/parsed/sent_align','c4-en-{}.json'.format(x)) for x in range(8)]
# word_outputs = [os.path.join('/data/home/tingchenfu/work1/data/parsed/word_align','c4-en-{}.json'.format(x)) for x in range(8)]
# switch_outputs = [os.path.join('/data/home/tingchenfu/work1/data/parsed/code_switch','c4-en-{}.json'.format(x)) for x in range(8)]
    docs=[]
    fin=open(input_file)
    for line in fin.readlines():
        doc=json.loads(line)['text']
        docs.append(doc)
    fin.close()

    fout_sent=open(sent_output_file,'w',encoding='utf-8')
    fout_word=open(word_output_file,'w',encoding='utf-8')
    fout_switch=open(switch_output_file,'w',encoding='utf-8')

    
    # for doc in tqdm(docs):
    #     sent_align,word_align,code_switch,doc_no =parsing_from_chinese((doc,-1))
    #     for sa in sent_align:
    #         fout_sent.write(json.dumps({'file_no':input_file.split('/')[-1],'doc_no': doc_no,'text':sa},ensure_ascii=False)+'\n')
    #     for wa in word_align:
    #         fout_word.write(json.dumps({'file_no':input_file.split('/')[-1],'doc_no': doc_no,'text':wa},ensure_ascii=False)+'\n')
    #     for cs in code_switch:
    #         fout_switch.write(json.dumps({'file_no':input_file.split('/')[-1],'doc_no': doc_no,'text':cs},ensure_ascii=False)+'\n') 
    parsing_fn = parsing_from_english if parsing_from == 'en' else parsing_from_chinese
    
    pool = Pool(cores)
    process_bar=tqdm(range(len(docs)))
    for sent_align,word_align,code_switch,doc_no in pool.imap_unordered(parsing_fn, zip(docs, list(range(len(docs))))):
        for sa in sent_align:
            fout_sent.write(json.dumps({'file_no':input_file.split('/')[-1],'doc_no': doc_no,'text':sa},ensure_ascii=False)+'\n')
        for wa in word_align:
            fout_word.write(json.dumps({'file_no':input_file.split('/')[-1],'doc_no': doc_no,'text':wa},ensure_ascii=False)+'\n')
        for cs in code_switch:
            fout_switch.write(json.dumps({'file_no':input_file.split('/')[-1],'doc_no': doc_no,'text':cs},ensure_ascii=False)+'\n')
        process_bar.update(1)
    pool.close()
    pool.join()