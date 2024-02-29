
def substring_mask(s1,s2,mask_id=0):
    m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)] 
    mmax = 0  
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i+1][j+1] = m[i][j] + 1
                if m[i+1][j+1] > mmax:
                    mmax = m[i+1][j+1]
                    p = j+1
    #print(mmax)
    for i in range(p-mmax,p):
        s2[i]=mask_id
    return s2
    #return mmax

def subsequence_mask(s1,s2,mask_id=0):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    path=[]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    x=m
    y=n
    while (x!=0) and (y!=0):
        if dp[x-1][y]==dp[x][y]:
            x-=1
            continue
        elif dp[x][y-1]==dp[x][y]:
            y-=1
            path.append(0)
            continue
        else:
            x-=1
            y-=1
            path.append(1)
    if len(path)<len(s2):
        path.extend([0]*(len(s2)-len(path)))
    path.reverse()
    for i in range(len(path)):
        if path[i]:
            s2[i]=mask_id
    return  s2

def overlap_mask(s1,s2,mask_id=0):
    wordset=set(s1)
    count=0
    for i in range(len(s2)):
        if s2[i] in wordset:
            s2[i]=mask_id

    return s2

lang_dict={
    'en':'English',
    'zh':'中文',
    'eng_Latn':'English',
    'cat_Latn':'Catalan',
    'pan_Guru':'Eastern Panjabi',
    'ibo_Latn':'Igbo',
    'tsn_Latn':'Tswana',
    'zho_Hans':'Chinese'
}


def template_for_ppl(examples,test_src,src_language,tgt_language):

    prompt=''
    suffix=''
    template="""
[{}]: {}
[{}]: {}"""
    prompt=suffix
    for example in examples:
        prompt+=template.format(lang_dict[src_language],example['src'],lang_dict[tgt_language],example['tgt'])
    prompt+=template.format(lang_dict[src_language],test_src,lang_dict[tgt_language],"")

    return prompt

def template_for_generation(examples,test_src,src_language,tgt_language):

    prompt=''
    suffix=''
    template="""[{}]: {} [{}]: {}"""
    prompt=suffix
    for example in examples:
        prompt+=template.format(lang_dict[src_language], example['src'], lang_dict[tgt_language], example['tgt'])
    prompt+=template.format(lang_dict[src_language], test_src, lang_dict[tgt_language], "")

    return prompt

# import os
# directory='/home/tingchen_fu/DialogStructure/dump_ubuntu/ensemums1'
# for file in os.listdir(directory):
#     if '25000' in file:
#         continue
#     path=os.path.join(directory,file)
#     os.system('rm '+path)
import datetime
import os
def update_argument(args,setting=None):
    MOUNT_DIR='/apdcephfs/share_916081/shared_info/tingchenfu'
    #  DATASET
    if args.dataset=='WMTnews21':
        if args.source_language == 'en' and args.target_language == 'zh':
            args.dev_src_file = [
                os.path.join(MOUNT_DIR,'Dataset/WMTnews/WMT2017.dev.enzh.src',), 
                os.path.join(MOUNT_DIR,'Dataset/WMTnews/WMT2017.test.enzh.src',), 
                os.path.join(MOUNT_DIR,'Dataset/WMTnews/WMT2018.test.enzh.src',), 
                os.path.join(MOUNT_DIR,'Dataset/WMTnews/WMT2019.test.enzh.src',),
            ]
            
            
            args.dev_tgt_file= [x.replace('src','tgt') for x in args.dev_src_file] 
            args.test_src_file = [
                os.path.join(MOUNT_DIR,'Dataset/WMTnews/WMT2021.test.enzh.src',), 
            ]
            args.test_tgt_file = [x.replace('src','tgt') for x in args.test_src_file] 
            
        elif args.source_language == 'zh' and args.target_language == 'en':
            args.dev_src_file = [
                os.path.join(MOUNT_DIR,'Dataset/WMTnews/WMT2017.dev.zhen.src',), 
                os.path.join(MOUNT_DIR,'Dataset/WMTnews/WMT2017.test.zhen.src',), 
                os.path.join(MOUNT_DIR,'Dataset/WMTnews/WMT2018.test.zhen.src',), 
                os.path.join(MOUNT_DIR,'Dataset/WMTnews/WMT2019.test.zhen.src',),
            ]
            args.dev_tgt_file= [x.replace('src','tgt') for x in args.dev_src_file] 
            args.test_src_file = [
                os.path.join(MOUNT_DIR,'Dataset/WMTnews/WMT2021.test.zhen.src',), 
            ]
            args.test_tgt_file = [x.replace('src','tgt') for x in args.test_src_file] 
        else:
            raise NotImplementedError

    elif args.dataset == 'flores200':
        args.dev_src_file = [os.path.join(MOUNT_DIR,'Dataset/flores200/dev/{}.dev'.format(args.source_language))]
        args.dev_tgt_file = [os.path.join(MOUNT_DIR,'Dataset/flores200/dev/{}.dev'.format(args.target_language))]
        args.test_src_file = [os.path.join(MOUNT_DIR,'Dataset/flores200/devtest/{}.devtest'.format(args.source_language))]
        args.test_tgt_file = [os.path.join(MOUNT_DIR,'Dataset/flores200/devtest/{}.devtest'.format(args.target_language))]
    
    elif args.dataset == 'MUSE':
        args.dev_src_file = [os.path.join(MOUNT_DIR,'Dataset/MUSE/muse_dict.{}{}.train.src'.format(args.source_language, args.target_language))]
        args.dev_tgt_file = [os.path.join(MOUNT_DIR,'Dataset/MUSE/muse_dict.{}{}.train.tgt'.format(args.source_language, args.target_language))]
        args.test_src_file = [os.path.join(MOUNT_DIR,'Dataset/MUSE/muse_dict.{}{}.test.src'.format(args.source_language, args.target_language))]
        args.test_tgt_file = [os.path.join(MOUNT_DIR,'Dataset/MUSE/muse_dict.{}{}.test.tgt'.format(args.source_language, args.target_language))]
        
    else:
        raise NotImplementedError
    

    # ### model_name_or_path
    # if args.model=='mBART-large-cc25':
    #     args.vocab_path = os.path.join(MOUNT_DIR,'PLM/mBART-large-cc25')
    #     if args.model_path is None:
    #         args.model_path = os.path.join(MOUNT_DIR,'PLM/mBART-large-cc25')
    # elif args.model == 'mT5-base':
    #     args.vocab_path = os.path.join(MOUNT_DIR,'PLM/mT5-base')
    #     if args.model_path is None:
    #         args.model_path = os.path.join(MOUNT_DIR,'PLM/mT5-base')
    # elif args.model == 'mGPT':
    #     args.vocab_path = os.path.join(MOUNT_DIR,'PLM/mT5-base')
    #     if args.model_path is None:
    #         args.model_path = os.path.join(MOUNT_DIR,'PLM/mGPT')
    # elif args.model == 'bloom-560m':
    #     args.vocab_path = os.path.join(MOUNT_DIR,'PLM/bloom-560m')
    #     if args.model_path is None:
    #         args.model_path = os.path.join(MOUNT_DIR,'PLM/bloom-560m')
    # elif args.model == 'bloomz-560m':
    #     args.vocab_path = os.path.join(MOUNT_DIR,'PLM/bloomz-560m')
    #     if args.model_path is None:
    #         args.model_path = os.path.join(MOUNT_DIR,'PLM/bloomz-560m')
    # elif args.model == 'bloom-7b':
    #     args.vocab_path = os.path.join(MOUNT_DIR,'PLM/bloom-7b1')
    #     if args.model_path is None:
    #         args.model_path = os.path.join(MOUNT_DIR,'PLM/bloom-7b1')
    # elif args.model == 'bloom-175b':
    #     args.vocab_path = os.path.join(MOUNT_DIR,'PLM/bloom-175b')
    #     if args.model_path is None:
    #         args.model_path = os.path.join(MOUNT_DIR,'PLM/bloom-175b')
    # elif args.model == 'bloom-3b':
    #     args.vocab_path = os.path.join(MOUNT_DIR,'PLM/bloom-3b')
    #     if args.model_path is None:
    #         args.model_path = os.path.join(MOUNT_DIR,'PLM/bloom-3b')
    # elif args.model == 'bloom-1b1':
    #     args.vocab_path = os.path.join(MOUNT_DIR,'PLM/bloom-1b1')
    #     if args.model_path is None:
    #         args.model_path = os.path.join(MOUNT_DIR,'PLM/bloom-1b1')
    # elif args.model == 'bloom-1b7':
    #     args.vocab_path = os.path.join(MOUNT_DIR,'PLM/bloom-1b7')
    #     if args.model_path is None:
    #         args.model_path = os.path.join(MOUNT_DIR,'PLM/bloom-1b7')
    # elif args.model == 'bloomz-7b':
    #     args.vocab_path = os.path.join(MOUNT_DIR,'PLM/bloomz-7b')
    #     if args.model_path is None:
    #         args.model_path = os.path.join(MOUNT_DIR,'PLM/bloomz-7b')
    # elif args.model == 'llama-7b':
    #     args.vocab_path = os.path.join(MOUNT_DIR,'PLM/llama-7b-hf')
    #     if args.model_path is None:
    #         args.model_path = os.path.join(MOUNT_DIR,'PLM/llama-7b-hf')
    # elif args.model == 'xglm-7b':
    #     args.vocab_path = os.path.join(MOUNT_DIR,'PLM/xglm-7b')
    #     if args.model_path is None:
    #         args.model_path = os.path.join(MOUNT_DIR,'PLM/xglm-7b')
    # else:
    #     raise NotImplementedError
    
    return args