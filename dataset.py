from torch.utils.data import Dataset
import json
import random
from tokenizers.pre_tokenizers import Whitespace
class MTPromptDataset(Dataset):
    # def __init__(self,data_files,debug,disturb=False) -> None:
    #     super(MTPromptDataset,self).__init__()
    #     self.examples=[]
    #     self.debug=debug
    #     self.disturb=disturb
    #     for data_file in data_files:
    #         f=open(data_file,encoding='utf-8')
    #         for line in f.readlines():
    #             record=json.loads(line)
    #             if isinstance(record['tgt'],str):
    #                 self.examples.append({'src':record['src'],'tgt':record['tgt']})
    #             else:
    #                 self.examples.append({'src':record['src'],'tgt':record['tgt'][0]})
    #             if self.debug and len(self.examples)==128:
    #                 break

    def __init__(self, src_files, tgt_files, debug, disturb=False) -> None:
        super(MTPromptDataset,self).__init__()
        self.examples=[]
        self.debug=debug
        self.disturb=disturb
        src_set = set()

        for src_file, tgt_file in zip(src_files,tgt_files):
            for line1, line2 in zip(open(src_file).readlines(), open(tgt_file).readlines()):
                line1=line1.strip('\n')
                line2=line2.strip('\n')
                if line1 == line2:
                    continue
                if line1 in src_set:
                    continue
                src_set.add(line1)
                self.examples.append({'src':line1,'tgt':line2})


    def __getitem__(self, index):
        if not self.disturb:
            return self.examples[index]['src'], self.examples[index]['tgt']
        else:
            return self.examples[(index+100)%len(self.examples)]['src'], self.examples[index]['tgt']

    def __len__(self):
        return len(self.examples)

    def get_example(self,src,tgt,example_fn,n_example,seed=0):
        examples=[]
        if example_fn=='rand':
            if seed ==0:
                examples = random.choices(self.examples,k=n_example)
            else:
                examples = self.examples[seed:(seed+n_example)%len(self.examples)]
        else:
            raise NotImplementedError

        return examples


    @staticmethod
    def collate_fn(batch):
        src_list=[item[0] for item in batch]
        tgt_list=[item[1] for item in batch]
        return src_list,tgt_list








class LanguageModelingDataset(Dataset):
    def __init__(self,data_files,n_data,min_corpus_length):
        super(LanguageModelingDataset,self).__init__()
        self.examples=[]
        pre_tokenizer = Whitespace()
        for data_file in data_files:
            f=open(data_file)
            for line in f.readlines():
                record=json.loads(line)
                if len(record['text'].split(' ')) < min_corpus_length:
                    continue
                self.examples.append(record['text'])

                if len(self.examples)==n_data:
                    break
                if len(self.examples) %10000 == 0 :
                    print("dataset loaded {}".format(len(self.examples)))
            if len(self.examples)==n_data:
                break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self,index):
        return self.examples[index]

    def collate_fn(batch):
        return batch