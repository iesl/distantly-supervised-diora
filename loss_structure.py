import torch
import torch.nn as nn
import numpy as np

from net_utils import *


class ConstrainedLoss(nn.Module):
    def __init__(self, batch_size, length, device, size, index, margin, print_match=False):
        super().__init__()
        self.batch_size = batch_size
        self.length = length
        self.device = device
        self.margin = margin
        self.print_match = print_match
        self.size = size
        self.index = index

        # print('batch_size, length, device, margin, size',batch_size, length, device, margin, size)



    def initial_chart(self, constraints=None):
        batch_size = self.batch_size
        length = self.length
        device = self.device
        dtype = torch.float32
        # spans: [[[start,length],[]],[[],[],...],...]
        offset_cache = self.index.get_offset(length)
        ncells = int(length * (1 + length) / 2)
        # print('length',length,'ncells',ncells)
        chart = torch.full((batch_size, ncells, 1), 0, dtype=dtype, device=device) # batch, levelxlength, 1
        subtract = torch.full((batch_size,1), 0, dtype=dtype, device=device) #batch, 1
        if constraints is None:
            return chart, subtract
        for idx, spans in enumerate(constraints):
            for span in spans:
                level = span[1]-1
                # assert level>0
                if level <=0:
                    continue
                pos = offset_cache[level] + span[0]
                chart[idx, pos] = 1000.0
                subtract[idx] += 1000.0
        return chart, subtract

    def batched_cky(self,s,chart,subtract=None):
        batch_size = self.batch_size
        length = self.length
        device = self.device
        size = self.size
        dtype = torch.float32
        for level in range(1, length):
            # print('level',level)
            B = batch_size
            L = length - level
            N = level
            batch_info = BatchInfo(
                batch_size=batch_size,
                length=length,
                size=size,
                level=level,
                phase='inside',
                )
            # if not subtract is None:
            #     print('!!chart',chart)
            ls, rs = get_inside_states(batch_info, chart, self.index, 1) #batch x L x N, 1
            xs = s[level] #B, L, N
            # if not subtract is None:
                # print('ls',ls,'rs',rs,'xs',xs)
            ps = ls.view(B,L,N,1) + rs.view(B,L,N,1) + xs
            # print('ps',ps,ps.shape,torch.max(ps,2))
            offset = self.index.get_offset(length)[level]
            # print('offset',offset,'L',L)
            chart[:,offset:offset+L] += torch.max(ps,2)[0] #B, L
        tree_scores = chart[:,-1]
        if not subtract is None:
            tree_scores -= subtract
        return tree_scores

    def marginal_loss(self, constr_s,pred_s):
        device = self.device
        dtype = torch.float32
        mask = torch.full(pred_s.shape, 1, dtype=dtype, device=device)
        mask[torch.abs(pred_s-constr_s)<0.001] = 0.0
        hinge = torch.clamp(self.margin + pred_s - constr_s, min=0)*mask
        # print(hinge)
        # print('mask',mask)
        # print('constr_s',constr_s)
        # print('pred_s',pred_s)
        if mask.sum() > 0.1:
            #print('mask is 0',hinge.sum())
            return hinge.sum()/mask.sum()
        return hinge.sum()

    def forward(self, score_components, constraints=None):
        constrained_chart, subtract = self.initial_chart(constraints)
        chart, _ = self.initial_chart()
        #print('constrained_chart',constrained_chart,'chart',chart,'subtract',subtract)

        s = {}
        for lvl in range(1,self.length):
            tmp = score_components[lvl] #{pos:[B,N]}
            score = []
            for pos in range(len(tmp)):
                score.append(tmp[pos])
            s[lvl] = torch.stack(score,1) #{lvl:[B,L,N]}

        constr_s = self.batched_cky(s,constrained_chart, subtract)
        pred_s = self.batched_cky(s,chart)
        #print('constr_s',constr_s,'pred_s',pred_s)
        loss = self.marginal_loss(constr_s,pred_s)


        return loss


class TreeStructure(nn.Module):
    name = 'tree_structure'

    def __init__(self, embeddings, input_size, size, cuda=False, margin=1, print=False, skip=False, weight=1.0):
        super().__init__()
        self.embeddings = embeddings
        self.margin = margin
        self.size = size
        self._cuda = cuda
        self.skip=skip
        self.weight = weight


    @classmethod
    def from_kwargs_dict(cls, context, kwargs_dict):
        kwargs_dict['embeddings'] = context['embedding_layer']
        kwargs_dict['cuda'] = context['cuda']
        return cls(**kwargs_dict)


    def forward(self, sentences, diora, info, embed=None):
        device = torch.cuda.current_device() if self._cuda else None
        batch_size, length = sentences.shape
        size = self.size


        score_components = diora.cache['inside_xs_components'] #{level:{pos:(B,N)}}
        constraints = info['constraints']
        index = diora.index
        tr_loss_func = ConstrainedLoss(batch_size, length, device, size, index,self.margin)
        tr_loss = tr_loss_func(score_components,constraints)

        loss = tr_loss * self.weight
        ret = {}
        ret[self.name + '_loss'] = loss

        return loss, ret

structure_class = TreeStructure