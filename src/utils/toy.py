import numpy as np
import torch
from torch import distributions as D
import matplotlib.pyplot as plt
from torch import nn
from torch.nn.parameter import Parameter


class GMM(nn.Module):
    def __init__(self, n=3, device=torch.cuda):
        super().__init__()
        self.device = device
        self.n = n
        self.p_s = Parameter(torch.tensor([1/n for _ in range(n)])).to(device)
        self.mu_s = Parameter(torch.stack([torch.randn(2)*2 for _ in range(n)], dim=0)).to(device)
        self.sigma_s = Parameter(torch.stack([torch.diag(torch.rand(2)) for _ in range(n)], dim=0)).to(device)
        self.update()
    
    def update(self):
        self.p_dist = D.Categorical(probs=self.p_s)
        self.dist_list = [
            D.MultivariateNormal(loc=self.mu_s[i],
                                             covariance_matrix=self.sigma_s[i])
            for i in range(self.n)
        ]

    def plot(self, samples=None, label=None, N=10000):
        self.update()
        if samples == None and label == None:
            samples, label = self.sample(N)
        plt.figure()
        samples = samples.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        plt.scatter(samples[:, 0], samples[:, 1], c=label, cmap="tab10")
        plt.show()

    def sample(self, N):
        self.update()
        samples = self.p_dist.sample([N]).reshape(N, 1, 1).repeat(1, 1, 2) # (N) -> (N, 1, 2)
        candidates = torch.stack([dist.sample([N]) for dist in self.dist_list], dim=1)
        out = torch.gather(candidates, dim=1, index=samples) # (100, 3, 2) -> (100, 1, 2)
        return out.squeeze(), samples.squeeze()[:, 0] # (N, 2), (N)

    def log_prob(self, samples):
        self.update()
        # (n, N)
        log_prob = torch.stack([self.p_dist.log_prob(torch.tensor(i, device=self.device))+self.dist_list[i].log_prob(samples) for i in range(self.n)], axis=0)
        return torch.logsumexp(log_prob, dim=0)
    
    def info(self):
        print("discrete p:", self.p_s)
        for i in range(self.n):
            print("-----------")
            print(self.mu_s[i])
            print(self.sigma_s[i])

class ToyGMM(GMM):
    def __init__(self, n, device):
        super().__init__(n=n, device=device)
        angles = [2*i*np.pi/self.n for i in range(self.n)]
        def mean(theta):
            return torch.tensor([np.cos(theta), np.sin(theta)], dtype=torch.float32).to(device)

        def get_covariance(theta):
            v1 = mean(theta)
            v2 = mean(theta+np.pi/2) # get vector perpend to v1
            Q = torch.stack([v1, v2], axis=1).to(device)
            D = torch.diag(torch.tensor([0.35, 0.08], dtype=torch.float32)**2).to(device)
            return Q@D@Q.T

        self.mu_s = Parameter(torch.stack([mean(x)  for x in angles], dim=0)).to(device)
        self.sigma_s = Parameter(torch.stack([get_covariance(x)  for x in angles], dim=0)).to(device)
        self.update()