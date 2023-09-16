#encoding=utf-8
import torch

embed_dim=20
#集成相似模型
class matrix_calculate(torch.nn.Module):
    def __init__(self,dict_index):
        super(matrix_calculate, self).__init__()
        self.embedding = torch.nn.Embedding(len(dict_index), embed_dim)
        self.cos = torch.nn.CosineSimilarity()
        self.linear=torch.nn.Linear(20,10)
        self.linear_2=torch.nn.Linear(10,1)
        self.parameter=torch.nn.Parameter(torch.Tensor([0.5,0.5]))
    def forward(self, DPTD_name_1, DPTD_name_2):
        DPTD_embed_1=self.embedding(torch.LongTensor([DPTD_name_1]))
        DPTD_embed_2 = self.embedding(torch.LongTensor([DPTD_name_2]))
        DPTD_embeds_1=self.linear(DPTD_embed_1)
        DPTD_embeds_2=self.linear(DPTD_embed_2)
        DPTD_embeds_2=torch.tanh(DPTD_embeds_2)#效果
        DPTD_embeds_3=self.linear_2(DPTD_embeds_2)
        DPTD_embeds=DPTD_embeds_3+DPTD_embeds_2
        cos=self.cos(DPTD_embeds_1,DPTD_embeds)
        dist=torch.dist(DPTD_embeds_1,DPTD_embeds)
        oula_loss=self.parameter[0]*cos+self.parameter[1]*dist
        return oula_loss,DPTD_embeds_1,DPTD_embeds
#余弦相似模型
class Cos_calculate(torch.nn.Module):
    def __init__(self,dict_index):
        super(Cos_calculate, self).__init__()
        self.embedding = torch.nn.Embedding(len(dict_index), embed_dim)
        self.cos = torch.nn.CosineSimilarity()
        self.linear=torch.nn.Linear(20,10)
    def forward(self, DPTD_name_1, DPTD_name_2):
        DPTD_embed_1=self.embedding(torch.LongTensor([DPTD_name_1]))
        DPTD_embed_2 = self.embedding(torch.LongTensor([DPTD_name_2]))
        DPTD_embeds_1=self.linear(DPTD_embed_1)
        DPTD_embeds_2=self.linear(DPTD_embed_2)
        loss=self.cos(DPTD_embeds_1,DPTD_embeds_2)
        return loss,DPTD_embeds_1,DPTD_embeds_2

#word2vec
class word_matrix_calculate(torch.nn.Module):
    def __init__(self,dict_index):
        super(word_matrix_calculate, self).__init__()
        self.embedding = torch.nn.Embedding(len(dict_index), embed_dim)
        self.cos = torch.nn.CosineSimilarity()
        self.linear=torch.nn.Linear(20,10)
        self.linear_2=torch.nn.Linear(10,1)
        self.linear_3=torch.nn.Linear(20,1)
        self.parameter=torch.nn.Parameter(torch.Tensor([0.5,0.5]))
    def forward(self, DPTD_name_1, DPTD_name_2,DPTD_word_1,DPTD_word_2):
        DPTD_embed_1=self.embedding(torch.LongTensor([DPTD_name_1]))
        DPTD_embed_2 = self.embedding(torch.LongTensor([DPTD_name_2]))
        DPTD_embed_3= self.embedding(torch.LongTensor(DPTD_word_1))
        DPTD_embed_4= self.embedding(torch.LongTensor(DPTD_word_2))
        DPTD_embeds_1=self.linear(DPTD_embed_1)
        DPTD_embeds_2=self.linear(DPTD_embed_2)
        DPTD_embeds_3=self.linear_3(DPTD_embed_3).reshape(1,10)
        DPTD_embeds_4=self.linear_3(DPTD_embed_4).reshape(1,10)
        DPTD_embed_1_news=DPTD_embeds_1+DPTD_embeds_3
        DPTD_embed_2_news=DPTD_embeds_2+DPTD_embeds_4
        DPTD_embeds_2_news=torch.tanh(DPTD_embed_2_news)#效果
        DPTD_embeds_3_news=self.linear_2(DPTD_embeds_2_news)
        DPTD_embeds=DPTD_embeds_3_news+DPTD_embeds_2_news
        cos=self.cos(DPTD_embed_1_news,DPTD_embeds)
        dist=torch.dist(DPTD_embed_1_news,DPTD_embeds)
        loss=self.parameter[0]*cos+self.parameter[1]*dist
        return loss,DPTD_embed_1_news,DPTD_embeds

class embed_calculate(torch.nn.Module):
    def __init__(self,dict_index):
        super(embed_calculate, self).__init__()
        self.embedding = torch.nn.Embedding(len(dict_index), embed_dim)
    def forward(self, DPTD_name_1, DPTD_name_2):
        DPTD_embed_1=self.embedding(torch.LongTensor([DPTD_name_1]))
        DPTD_embed_2=self.embedding(torch.LongTensor([DPTD_name_2]))
        return DPTD_embed_1,DPTD_embed_2
class word_calculate(torch.nn.Module):
    def __init__(self,dict_index):
        super(word_calculate, self).__init__()
        self.embedding = torch.nn.Embedding(len(dict_index), embed_dim)
        self.linear = torch.nn.Linear(20, 10)
        self.linear_3 = torch.nn.Linear(20, 1)
    def forward(self, DPTD_name_1, DPTD_name_2, DPTD_word_1, DPTD_word_2):
        DPTD_embed_1 = self.embedding(torch.LongTensor([DPTD_name_1]))
        DPTD_embed_2 = self.embedding(torch.LongTensor([DPTD_name_2]))
        DPTD_embed_3 = self.embedding(torch.LongTensor(DPTD_word_1))
        DPTD_embed_4 = self.embedding(torch.LongTensor(DPTD_word_2))
        DPTD_embeds_1 = self.linear(DPTD_embed_1)
        DPTD_embeds_2 = self.linear(DPTD_embed_2)
        DPTD_embeds_3 = self.linear_3(DPTD_embed_3).reshape(1, 10)
        DPTD_embeds_4 = self.linear_3(DPTD_embed_4).reshape(1, 10)
        DPTD_embed_1_news = DPTD_embeds_1 + DPTD_embeds_3
        DPTD_embed_2_news = DPTD_embeds_2 + DPTD_embeds_4
        return DPTD_embed_1_news, DPTD_embed_2_news

class word2sep_matrix_calculate(torch.nn.Module):
    def __init__(self,dict_index):
        super(word2sep_matrix_calculate, self).__init__()
        self.embedding = torch.nn.Embedding(len(dict_index), embed_dim)
        self.cos = torch.nn.CosineSimilarity()
        self.linear=torch.nn.Linear(20,10)
        self.linear_2=torch.nn.Linear(10,1)
        self.linear_3=torch.nn.Linear(20,1)
        self.parameter=torch.nn.Parameter(torch.Tensor([0.5,0.5]))
    def forward(self,DPTD_word_1,DPTD_word_2):
        DPTD_embed_1=self.embedding(torch.LongTensor(DPTD_word_1))
        DPTD_embed_2 = self.embedding(torch.LongTensor(DPTD_word_2))
        DPTD_embeds_1=self.linear_3(DPTD_embed_1).reshape(1,10)
        DPTD_embeds_2=self.linear_3(DPTD_embed_2).reshape(1,10)
        DPTD_embeds_2_news=torch.tanh(DPTD_embeds_2)#效果
        DPTD_embeds_3_news=self.linear_2(DPTD_embeds_2_news)
        DPTD_embeds=DPTD_embeds_3_news+DPTD_embeds_2_news
        cos=self.cos(DPTD_embeds_1,DPTD_embeds)
        dist=torch.dist(DPTD_embeds_1,DPTD_embeds)
        loss=self.parameter[0]*cos+self.parameter[1]*dist
        return loss,DPTD_embeds_1,DPTD_embeds



def result_matrix(random_matrix, wr_matirx, dict_1):
    result = dict_1 + torch.mm(wr_matirx.T, random_matrix)
    return result