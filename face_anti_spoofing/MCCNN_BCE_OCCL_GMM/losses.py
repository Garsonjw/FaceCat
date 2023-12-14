import torch
import torch.nn as nn
import numpy as np 

from torch.nn import functional as F

import torch.nn

from torch.autograd import Variable


class OCCL(torch.nn.Module):
	"""
	One Class Contrastive Loss Function.

	"""

	def __init__(self, margin=3.0, feat_dim=10, alpha=0.1,device='cuda:0', center_adapt=True):
		super(OCCL, self).__init__()
		self.device=device
		self.margin = margin
		self.feat_dim = feat_dim
		self.center_adapt=center_adapt
		if self.center_adapt:
			self.center = torch.randn(1, self.feat_dim, requires_grad=False).to(self.device)
		else:
			self.center = torch.zeros(1, self.feat_dim, requires_grad=False).to(self.device)

		self.alpha =alpha

	def forward(self, x, label, is_training=True):
		
		# center updation
		batch_size = x.size(0)

		indices = np.where(label.cpu().numpy()==1)[0] #bonafide Note : The ground truth conventions!

		if len(indices)!=0 and is_training and self.center_adapt:
			self.center = self.center + self.alpha * torch.mean(x[indices].detach() - self.center.expand(len(indices),-1))

		expanded_centers = self.center.expand(batch_size, -1) # batchxfeatdim

		euclidean_distance = F.pairwise_distance(x, expanded_centers)

		bonafide_loss=(label) * torch.pow(euclidean_distance, 2)
		
		attack_loss =(1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)

		loss = torch.mean(bonafide_loss+attack_loss)			  

		return loss

