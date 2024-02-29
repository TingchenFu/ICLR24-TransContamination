import re
import json
import sys
sys.path.append('/data/home/tingchenfu/work1')
from data import thulac
thu = thulac.thulac(seg_only=True)




no_punc = re.compile(r'[^\w\s]')
no_number = re.compile(r'\d+')
no_extra_space = re.compile(r'\s+')
only_character= re.compile(r'[^\u4e00-\u9fff]+')
# import regex


def norm_en(ori_string):
    lower_string = ori_string.lower()
    no_number_string = no_number.sub('',lower_string)
    no_punc_string = no_punc.sub('',no_number_string)
    # remove all punctuation except words and space
    # remove white spaces
    no_exspace_string = no_extra_space.sub(' ', no_punc_string)

    return no_exspace_string


def norm_zh(ori_string):
    text = only_character.sub('',ori_string)
    text = no_extra_space.sub('', text).strip()
    return thu.cut(text,text=True)

count = 0 
fout=open('/apdcephfs/share_916081/shared_info/tingchenfu/work1/data/pure_1m/zh_1k_text','w')
for line in open('/apdcephfs/share_916081/shared_info/tingchenfu/work1/data/pure_1m/zh_1m').readlines():
    text = json.loads(line)['text']
    text = norm_zh(text)
    fout.write(text.replace('\n','')+'\n')
    count+=1
    if count==1000:
        break