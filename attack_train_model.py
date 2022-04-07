import torch
from torch import nn


import torch
import torch.nn as nn


# FGM
class FGM:
    def __init__(self, model: nn.Module, eps=1.):
        self.model = (
            model.module if hasattr(model, "module") else model
        )
        self.eps = eps
        self.backup = {}

    # only attack word embedding
    def attack(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm and not torch.isnan(norm):
                    r_at = self.eps * param.grad / norm
                    param.data.add_(r_at)
                else:
                    print('norm is torch nan')

    def restore(self, emb_name='word_embeddings'):
        for name, para in self.model.named_parameters():
            if para.requires_grad and emb_name in name:
                assert name in self.backup
                para.data = self.backup[name]

        self.backup = {}
# class FGM:
#     def __init__(self, eps=1.) -> None:
#         self.backup = {}
#         self.eps = eps
    
#     def attack(self, model: nn.Module):
#         for name, weights in model.named_parameters():
#             if 'word_embeddings' in name:
#                 self.backup[name] = weights.data.clone()
#                 norm = torch.norm(weights.grad)
#                 r_t = self.eps * weights.grad / norm
#                 weights.data.add_(r_t)
        
#     def restore(self, model: nn.Module):
#         for name, weights in model.named_parameters():
#             if 'word_embedings' in name:
#                 assert name in self.backup, '{} not in backup weight data'.format(name)
#                 weights.data = self.backup[name]