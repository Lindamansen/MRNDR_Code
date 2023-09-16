#encoding=utf-8
import torch

embed_dim=20
class word2Sequence:
    UNK_tag="UNK"
    PAD_tag="PAD"
    UNK=0
    PAD=1
    def __init__(self):
        self.dict={self.UNK_tag:self.UNK,self.PAD_tag:self.PAD}
        self.count={}
    def fit(self,words):
        for word in words:
            self.count[word]=self.count.get(word,0)+1
    def build_vocab(self,min=5,max=None,max_features=None):
        if min is not None:
            self.count={word:value for word,value in self.count.items() if value>min}
        if max is not None:
            self.count={word:value for word,value in self.count.items() if value<max}
        if max_features is not None:
            temp=sorted(self.count.items(),key=lambda x:x[-1],reverse=True)[:max_features]
            self.count=dict(temp)
        for word in self.count:
            self.dict[word]=len(self.dict)
        self.inverse_dict=dict(zip(self.dict.values(),self.dict.keys()))
    def transform(self,words,max_len=None):
        if max_len is not None:
            if max_len > len(words):
                words=[self.PAD_tag]*(max_len-len(words))
            if max_len < len(words):
                words=words[:max_len]
        return [self.dict.get(word,self.UNK) for word in words]
    def inverse_transform(self,indices):
        return [self.inverse_dict.get(idx) for idx in indices]
    def __len__(self):
        return len(self.dict)

def word_code(dict,name):
    ws = word2Sequence()
    ws.fit(dict)
    ws.build_vocab(min=0)
    count = ws.transform(name)
    # count_inv=ws.inverse_transform(count)
    # print(count_inv)
    return count