#encoding=utf-8

def Mrr(S,rank_all):
    rank_sum=0
    for rank in rank_all:
        if rank==0:
            rank=1
            ranki=1/rank
            rank_sum+=ranki
        else:
            ranki = 1 / rank
            rank_sum += ranki
    Mrr_score=rank_sum/S
    return Mrr_score

def His(S,N,rank_all):
    rank_sum = 0
    for rank in rank_all:
        if rank<=N:
            if rank==0:
                rank_sum +=1
            else:
                rank_sum +=1
        else:
            rank_sum +=0
    His_score = rank_sum/S
    return His_score

