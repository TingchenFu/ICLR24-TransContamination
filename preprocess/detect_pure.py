import json
import re
import sys
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm

#pattern=re.compile(r'[A-Za-z0-9 _\,\.\;\:\'\"\/\?\!\-#\&]*[a-zA-Z]{3,}[A-Za-z0-9 _\,\.\;\:\'\"\/\?\!\-#\&]*')

# filter_pattern=re.compile(r'\\u\d\d\d\d|\\b|\\x\d\d')

chinese_pattern = re.compile(r'[\u2e80-\u9fff]+')
english_pattern = re.compile(r'[a-zA-Z]')



def pure_english(inputs):
    '''
    filter out chinese charactersitics from the english corpus to get pure english
    '''
    pured=[]
    doc,doc_no=inputs
    try:
        text = json.loads(doc)['text']
    except:
        text= doc
    lines=text.split('\n')
    for line in lines:
        # try:
        #     detected=ftlangdetect.detect(line)
        #     if  detected['lang'] =='en' and detected['score'] > 0.8:
        #         pured.append(line)
        # except:
        #     pass
        filtered = line
        filtered = re.sub(chinese_pattern,'',filtered)
        pured.append(filtered)
    return '\n'.join(pured), doc_no


def pure_chinese(inputs):
    '''
    filter out English charactersitics from the english corpus to get pure chiense
    '''
    replace_list=[u'\u0005',u'\u0006',u'\u0007',u'\u0001',u'\u0002',u'\u0003',u'\u0004',u'\u0008']
    
    pured=[]
    doc,doc_no=inputs
    try:
        text = json.loads(doc)['text']
    except:
        text= doc
    lines=text.split('\n')
    for line in lines:
        for ch in replace_list:
            line = line.replace(ch,'')
        line = re.sub(english_pattern,'',line)
        pured.append(line)        
    return '\n'.join(pured), doc_no

if __name__ =='__main__':

    try:
        input_file=sys.argv[1]
        output_file=sys.argv[2]
        parsing_from = sys.argv[3]
        debug=False
    
    except:
        input_file='/data/home/tingchenfu/work1/debug_zh.json'
        output_file='/data/home/tingchenfu/work1/debug_pure'
        parsing_from = 'zh'
        debug=True
    
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

    fout=open(output_file,'w',encoding='utf-8')
    # fout_sent=open(sent_output_file,'w',encoding='utf-8')
    # fout_word=open(wo
    # rd_output_file,'w',encoding='utf-8')
    # fout_switch=open(switch_output_file,'w',encoding='utf-8')
    
    # for doc in tqdm(docs):
    #     sent_align,word_align,code_switch,doc_no =parsing_from_chinese((doc,-1))
    #     for sa in sent_align:
    #         fout_sent.write(json.dumps({'file_no':input_file.split('/')[-1],'doc_no': doc_no,'text':sa},ensure_ascii=False)+'\n')
    #     for wa in word_align:
    #         fout_word.write(json.dumps({'file_no':input_file.split('/')[-1],'doc_no': doc_no,'text':wa},ensure_ascii=False)+'\n')
    #     for cs in code_switch:
    #         fout_switch.write(json.dumps({'file_no':input_file.split('/')[-1],'doc_no': doc_no,'text':cs},ensure_ascii=False)+'\n') 
    parsing_fn = pure_english if parsing_from == 'en' else pure_chinese

    # pool = Pool(max(cores,32))
    # process_bar=tqdm(range(len(docs)))
    # for pured,doc_no in pool.imap_unordered(parsing_fn, zip(docs, list(range(len(docs))))):
    #     for pu in pured:
    #         fout.write(json.dumps({'file_no':input_file.split('/')[-1],'doc_no': doc_no,'text':pu},ensure_ascii=False)+'\n')
    #     process_bar.update(1)
    # pool.close()
    # pool.join()


    doc_no=0
    for doc in tqdm(docs):
        pured,_ = parsing_fn((doc,doc_no))
        if pured:
            fout.write(json.dumps({'file_no':input_file.split('/')[-1],'doc_no': doc_no,'text':pured},ensure_ascii=False)+'\n')
        doc_no+=1