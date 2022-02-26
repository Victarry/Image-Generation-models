import torch
from torch import nn
import torch.distributions as D
import torch.nn.functional as F

class GaussianDistribution(nn.Module):
    def __init__(self, mu):
        super().__init__()
        self.mu = mu
        self.dist = D.Normal(mu, torch.ones_like(mu))
    
    def prob(self, target):
        p_x = self.dist.log_prob(target).sum(dim=[1,2,3])
        return p_x
        
    def sample(self, mu):
        return mu

class BernoulliDistribution(nn.Module):
    def __init__(self, input_normalized=True):
        super().__init__()
        self.input_normalized = input_normalized
    
    def forward(self, logits, target):
        if self.input_normalized:
            target = (target+1)/2
        prob = -F.binary_cross_entropy_with_logits(logits, target, reduction='none').sum([1, 2, 3])
        return prob

    def sample(self, logits):
        # NOTE: Actually, sampling from bernoulli will cause sharp artifacts.
        # dist = distributions.Bernoulli(logits=logits)
        # imgs = dist.sample()
        imgs = torch.sigmoid(logits)
        if self.input_normalized:
            imgs = imgs*2-1
        return imgs