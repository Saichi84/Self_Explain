# def load_data(input_dir:str,output:str):
import torch
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModel, get_cosine_schedule_with_warmup
from torch.utils.data import Dataset
from demo.collate_func import collate_to_max_length
# tokenizer = AutoTokenizer.from_pretrained('roberta-base')
from torch.utils.data.dataloader import DataLoader

import json as js
def read_train_line(line:str):
    data=line.replace("\n","").split("\t",6)
    target_id=data[0]
    prior_id=data[1]
    tree=js.loads(data[2])
    target_claim=js.loads(data[3])
    rel_claim_keys=[int(i) for i in data[4].split(',')]
    prior_passage=js.loads(data[5])
    cate=int(data[6])
    
    
    return target_id, prior_id, tree, target_claim, rel_claim_keys, prior_passage, cate

def gen_set_of_claim(tree:dict):
    claim_list=[[i] for i in tree.keys()]
    tree_list=[tree]
    # deep=3
    while tree_list:
        cur_tree=tree_list.pop(0)
        for father_node in cur_tree.keys():
            if not cur_tree[father_node]:
                continue
            tree_list.append(cur_tree[father_node])
            for son_node in cur_tree[father_node]:
                for cs_i in claim_list.copy():
                    if father_node in cs_i:
                        tmp=cs_i.copy()
                        tmp.append(son_node)
                        # case_1: the len of set of claim is no more than 4 (level 3). 
                        if len(tmp)>4:
                            continue
                        claim_list.append(tmp)
                        # print(claim_list)

    return claim_list

class Patent_Dataset(Dataset):

    def __init__(self, data_path,mode='train', max_token_len: int = 128, sample = 500):
        self.data_path = data_path
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.model=AutoModel.from_pretrained('distilroberta-base')
        self.max_token_len = max_token_len
        self.sample = sample
        self.mode=mode
        self._prepare_data()


    def _prepare_data(self):
        # data = pd.read_csv(self.data_path)
        # data['unhealthy'] = np.where(data['healthy'] == 1, 0, 1)
        # if self.sample is not None:
        #   unhealthy = data.loc[data[attributes].sum(axis=1) > 0]
        #   clean = data.loc[data[attributes].sum(axis=1) == 0]
        #   self.data = pd.concat([unhealthy, clean.sample(self.sample, random_state=7)])
        # else:
        #   self.data = data
        self.data=[]
        with open(self.data_path,'r',encoding='utf-8') as f:
        #     count=0
            for line in f:
                data_line=list(self.load_data(line))
                self.data.append(data_line)
                # data_line=[]
        #         inf=line.split("\t\t")
        #         data_line.append(inf[0].split("\t"))
        #         data_line.append(inf[1].split("\t"))
        #         data_line.append(bool(inf[2]))
        #         self.data.append(data_line.copy())
        #         # print(len(data_line),data_line[2],inf[2])
        #         # count+=1
        #         # if count>5:
        #         #   break
    def __len__(self):
        return len(self.data)

    def get_representation(self,sentence:str):
        tokens = self.tokenizer.encode_plus(sentence,
                                            add_special_tokens=True,
                                            return_tensors='pt',
                                            truncation=True,
                                            padding='max_length',
                                            max_length=self.max_token_len,
                                            return_attention_mask = True)

        out=self.model(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask)
        rel=torch.mean(out.last_hidden_state, 1)
        return rel

    def __getitem__(self, index):
        item = self.data[index]
        if self.mode=='train':
            target_represent=torch.cat([self.get_representation(x) for x in item[0]])
            prior_represent=torch.cat([self.get_representation(x) for x in item[1]])

            label=torch.LongTensor([item[2]])
            return {'target_represent':target_represent,'prior_represent':prior_represent,'label':label}

        else:
            target_idx,claim_set_idx_list,target_claim=item[0]
            prior_idx,prior_passage_indx_list,prior_passage=item[1]

            target_represent=torch.cat([self.get_representation(x) for x in item[2][0]])
            prior_represent=torch.cat([self.get_representation(x) for x in item[2][1]])
            # print(float(item[2]))
            # labels=torch.FloatTensor([float(item[2])])
            label=torch.LongTensor([item[2][2]])


        return {'target_idx':target_idx,'claim_set_idx_list':claim_set_idx_list,'target_claim':target_claim,
                'prior_idx':prior_idx,'prior_passage_indx_list':prior_passage_indx_list,'prior_passage':prior_passage,
                'target_represent':target_represent,'prior_represent':prior_represent,'label':label}

    def load_data(self,line:str):
        # with open (data_file,encoding='utf-8') as f:
            # count=0
            # for line in f:
                # count+=1
                # example=read_train_line(line)
        target_id, prior_id, tree, target_claim, rel_claim_keys, prior_passage, cate=read_train_line(line.replace("\n",""))
    
        # Create set of claim by dep-tree
        claim_set_idx_list=gen_set_of_claim(tree)

        # Get value for each set of claim
        claim_set=[]
        for claim_set_ids in claim_set_idx_list:
            # claim_set_ti_key="&".join(claim_set_ids)
            claim_set_ti_value=""
            for i in claim_set_ids:
                claim_set_ti_value+=target_claim[i]
            claim_set.append(claim_set_ti_value)

        # Split value of prior_passage
        # Need to be changed (in order to optimize the data mining)
        prior_passage_indx_list=[]
        prior_passage_list=[]
        for element in prior_passage.keys():
            prior_passage_list.extend(prior_passage[element])
            prior_passage_indx_list.extend([[element,i+1] for i in range(len(prior_passage[element]))])
            # if element=="wp":
            #     prior_passage_list.extend(prior_passage[element])
            #     prior_passage_indx_list.extend([element+'-Passage: '+str(i+1) for i in range(len(prior_passage[element]))])
            # else:
            #     prior_passage_list.append(prior_passage[element])
            #     prior_passage_indx_list.append(element)
        
    
        # Convert list_idx to string
        # claim_set_idx_list=['Claim '+" & ".join(claim_set_idx) for claim_set_idx in claim_set_idx_list]

        # Limit the len of claim_set and rel-passage
        if len(claim_set)>40:
            claim_set=claim_set[:40]
        if len(prior_passage_list)>40:
            prior_passage_list=prior_passage_list[:40]
            
        if self.mode=='train':
            return (claim_set,prior_passage_list,cate)

        return (target_id,claim_set_idx_list,target_claim),(prior_id,prior_passage_indx_list,prior_passage),(claim_set,prior_passage_list,cate)


