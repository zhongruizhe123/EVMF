######################################################################################
# The implementation relies on http://nlp.seas.harvard.edu/2018/04/03/attention.html #
######################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothing(nn.Module):
    
    def __init__(self, smoothing, pad_idx):
        super(LabelSmoothing, self).__init__()
        self.smoothing = smoothing          #0.7一种随机的超参数
        self.pad_idx = pad_idx
        
    def forward(self, pred, target): # pred (B, S, V), target (B, S)
        # Note: preds are expected to be after log
        B, S, V = pred.shape
        # print("predloss1.shape",pred.shape)         #predloss.shape torch.Size([64, 30, 10173])
        # print("targetloss1.shape", target.shape)       #targetloss.shape torch.Size([64, 30])
        # (B, S, V) -> (B * S, V); (B, S) -> (B * S)
        pred = pred.contiguous().view(-1, V)            #叠加前两个维度
        target = target.contiguous().view(-1)
        # print("pred.shape2",pred.shape)                 #pred.shape2 torch.Size([2112, 10173])
        # print("target.shape2",target.shape)
        dist = self.smoothing * torch.ones_like(pred) / (V - 2)
        # print("dist", dist.shape)
        # add smoothed ground-truth to prior (args: dim, index, src (value))
        dist.scatter_(1, target.unsqueeze(-1).long(), 1-self.smoothing)

        # make the padding token to have zero probability
        dist[:, self.pad_idx] = 0
        # ?? mask: 1 if target == pad_idx; 0 otherwise
        mask = torch.nonzero(target == self.pad_idx)
        
        if mask.sum() > 0 and len(mask) > 0:
            # dim, index, val
            dist.index_fill_(0, mask.squeeze(), 0)
            # print("dist2", dist.shape)
        return F.kl_div(pred, dist, reduction='sum')


class SimpleLossCompute(object):

    def __init__(self, criterion, lr_scheduler):
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler

    def __call__(self, pred, target, normalize):
        loss = self.criterion(pred, target) / normalize
        # print("loss", loss)
        loss.backward(retain_graph=True)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            self.lr_scheduler.optimizer.zero_grad()

        return loss * normalize
    
class SimpleLossCompute_memory(object):
    
    def __init__(self, criterion, lr_scheduler):  
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.regression_loss = torch.nn.MSELoss(size_average=False)  # none 不求平均 # 默认为mean #sum

    def __call__(self, pred, out_memory, new_memory_video, memory_audio, target, normalize):
        loss1 = self.criterion(pred, target) / normalize
        # loss2 = self.criterion(out_memory, target) / normalize
        # loss3 = self.regression_loss(memory_audio, new_memory_video) / normalize * 2
        print("loss1", loss1, "loss2", loss2, "loss3", loss3, "normalize:", normalize)
        loss = loss1
        loss.backward()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            self.lr_scheduler.optimizer.zero_grad()
        
        return loss * normalize
