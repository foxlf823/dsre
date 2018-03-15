import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import utils
import math
import numpy as np
import copy

class PCNN(nn.Module):
        
    def __init__(self, dimensionC, relationTotal, dimensionWPE, dimension, 
                 window, wordTotal, PositionTotalE1, PositionTotalE2, wordVec, dropout):
        super(PCNN, self).__init__()
        
        con = math.sqrt(6.0/(3*dimensionC+relationTotal))
        con1 = math.sqrt(6.0/((dimensionWPE+dimension)*window))
        
        self.word_emb = nn.Embedding(wordTotal, dimension)
        self.word_emb.weight = nn.Parameter(wordVec)
        
        self.position_e1_emb = nn.Embedding(PositionTotalE1, dimensionWPE)
        self.position_e1_emb.weight = nn.Parameter(torch.from_numpy(np.random.uniform(-con1, con1, size=(PositionTotalE1, dimensionWPE))).float())
        
        self.position_e2_emb = nn.Embedding(PositionTotalE2, dimensionWPE)
        self.position_e2_emb.weight = nn.Parameter(torch.from_numpy(np.random.uniform(-con1, con1, size=(PositionTotalE2, dimensionWPE))).float())
        
        self.d = dimension+2*dimensionWPE
        self.pad_len = window-1
        self.conv = nn.Conv2d(1, dimensionC, (window, self.d), (1, 1), (self.pad_len, 0))
        self.conv.weight = nn.Parameter(torch.from_numpy(np.random.uniform(-con1, con1, size=(dimensionC, 1, window, self.d))).float())
        self.conv.bias = nn.Parameter(torch.from_numpy(np.random.uniform(-con, con, size=(dimensionC))).float())
        
        self.output = nn.Linear(3*dimensionC, relationTotal, False)
        self.output.weight = nn.Parameter(torch.from_numpy(np.random.uniform(-con, con, size=(relationTotal, 3*dimensionC))).float())
        
        self.dropout = dropout
        
    def forward(self, word, pos1, pos2, piece_wise, b_drop):
        batch = word.size(0)
        sent_len = word.size(1)
        
        emb_w = self.word_emb(word) # (batch, sent_len, word_dim)
        emb_pos1 = self.position_e1_emb(pos1) # (batch, sent_len, position_dim)
        emb_pos2 = self.position_e2_emb(pos2)
        
        q = torch.cat((emb_w, emb_pos1, emb_pos2), 2) # (batch, sent_len, self.d)
        
        q = q.view(batch, 1, -1, self.d) # (batch, 1, sent_len, self.d)
        
        c = self.conv(q) # (batch, dimensionC, sent_len+window-1, 1)
        c = c.squeeze(dim=-1)
        
        # the length after conv is 'window-1' longer than sent_len
        new_piece_wise = copy.copy(piece_wise)
        new_piece_wise[-1] += self.pad_len
        
        c1, c2, c3 = utils.size_splits(c, new_piece_wise, 2)
        p1 = F.max_pool1d(c1, new_piece_wise[0]) # (batch, dimensionC, 1)
        p2 = F.max_pool1d(c2, new_piece_wise[1])
        p3 = F.max_pool1d(c3, new_piece_wise[2])
        
        p = torch.cat((p1.squeeze(-1),p2.squeeze(-1),p3.squeeze(-1)), dim=1)
        
        g = F.tanh(p)
        
        if b_drop:
            g = F.dropout(g, self.dropout, self.training)
            
        o = self.output(g)
         
        return o
