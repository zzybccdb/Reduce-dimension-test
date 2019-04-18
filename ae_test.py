import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

class AutoencoderTest(nn.Module):
	name = 'AutoencoderTest'
	def __init__(self, input_dim, lw):
		super(AutoencoderTest, self).__init__()

		self.input_dim = input_dim
		print('input_dim:', self.input_dim)
		N_EMBED = 32
		self.set_network(1, 1)

		self.mse = nn.MSELoss()
		self.mse_each = nn.MSELoss(reduction='none')

		self.bce = nn.BCELoss()
		self.bce_each = nn.BCELoss(reduction='none')

		self.criterion = self.mse
		self.criterion_each = self.mse_each

		self.lw = lw
		self.alpha = nn.Parameter(torch.Tensor([1.0]))

		# self.high_low_dis = None

	def set_network(self, input_window, output_window):
		
		self.encoder_net = nn.Sequential(
			nn.Linear(self.input_dim * input_window, 128),
			nn.Tanh(),
			nn.Linear(128, 64),
			nn.Tanh(),
			nn.Linear(64, 32),
			nn.Tanh(),
			nn.Linear(32, 8),
			nn.Tanh(),
			nn.Linear(8, 2))
		# self.encoder_net = nn.Sequential(
		# 	nn.Linear(input_dim, 256),
		# 	nn.Tanh(),
		# 	nn.Linear(256, 128),
		# 	nn.Tanh(),
		# 	nn.Linear(128, 64),
		# 	nn.Tanh(),
		# 	nn.Linear(64, N_EMBED),
		# )

		self.decoder_net = nn.Sequential(
			nn.Linear(2, 8),
			nn.Tanh(),
			nn.Linear(8, 32),
			nn.Tanh(),
			nn.Linear(32, 64),
			nn.Tanh(),
			nn.Linear(64, 128),
			nn.Tanh(),
			nn.Linear(128, self.input_dim * output_window),
			nn.Sigmoid(),)

	def encoder(self, x):
		return self.encoder_net(x)

	def decoder(self, z):
		return self.decoder_net(z)

	def latent(self, x):
		return self.encoder(x)

	def forward(self, x):
		z = self.encoder(x)
		output = self.decoder(z)
		return output

	def reconstruct(self, x):
		return self.forward(x)

	def distance_matrix_Z(self,data,eps=1e-8):
		x1 = data.unsqueeze(1).repeat(1, data.shape[0], 1)
		x2 = data.unsqueeze(0)
		dist = (x1 - x2).pow(2).sum(2)
		dist = (dist+eps).sqrt()
		# get the row size 
		r = data.shape[0] 
		mask = torch.triu(torch.ones([r,r]),diagonal=1)
		
		return dist[mask==1]

	def distance_matrix_Ｘ(self,data,g_mean,eps=1e-8):
		x1 = data.unsqueeze(1).repeat(1, data.shape[0], 1)
		x2 = data.unsqueeze(0)
		# w[0][0] = 1
		# w[0][1] = 1
		# w[0][2] = 1
		# w[0][3] = 10
		# w[0][4] = 10
		dist = (x1 - x2).pow(2)
		dist = dist.sum(2)
		# dist = (x1 - x2).pow(2).sum(2)
		dist = (dist+eps).sqrt()
		# get the row size 
		r = data.shape[0] 
		
		mask = torch.triu(torch.ones([r,r]),diagonal=1)
		# dist = torch.triu(dist,diagonal=1)
		dist[mask==1] = (dist[mask==1])/g_mean
		flat = dist[mask==1]
		
		return flat

	def loss(self, x, y, g_mean):
		"""
		x: shape = BATCH_SIZE x self.input_dim
		z: shape = BATCH_SIZE x 2
		"""
		# print("Current loss weight is:",self.lw)
		z = self.encoder(x)
		recons = self.decoder(z)
		xd = self.distance_matrix_Ｘ(data=x, g_mean=g_mean)
		zd = self.distance_matrix_Z(z)
		zd_s = 1 * zd

		#map_dist = torch.abs(xd-zd_s).mean()
		map_dist = F.mse_loss(zd_s,xd)
		map_dist_s = self.lw * map_dist
		rec_loss = self.criterion(recons, y)

		return rec_loss, map_dist_s, map_dist

	def loss_each(self, x):
		recons = self.forward(x)
		return self.criterion_each(recons, x)
