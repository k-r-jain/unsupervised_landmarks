import collections

import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision
# from torchsummary import summary
import math


device = 'cuda:0'
class BaseEncoder(nn.Module):
	def __init__(self, input_channels = 3, init_channels = 32, growth_per_block = 2):
		super(BaseEncoder, self).__init__()
		self.input_channels = input_channels
		self.init_channels = init_channels
		self.growth_per_block = growth_per_block
		self.net = self.network(self.input_channels, self.init_channels, self.growth_per_block)

	def conv_unit(self, in_ch, out_ch, kernel_size = 3, stride = 1, padding = 0, batch_norm = True, activation_relu = True):
		modules = []
		modules.append(nn.Conv2d(in_channels = in_ch, out_channels = out_ch, kernel_size = kernel_size, stride = stride, padding = padding))
		# Paper uses BN before ReLU
		if batch_norm:
			modules.append(nn.BatchNorm2d(num_features = out_ch))
		if activation_relu:
			modules.append(nn.ReLU())
		return nn.Sequential(*modules)


	def network(self, input_channels, init_channels, growth_per_block):
		# Reduces 128x128 to 16x16. BN + ReLU is applied to the last layer as well in paper
		modules = []
		modules.append(self.conv_unit(input_channels, init_channels, kernel_size = 7, stride = 1, padding = 1))
		modules.append(self.conv_unit(init_channels, init_channels, kernel_size = 3, stride = 1, padding = 1))

		for _ in range(3):
			prev_channels = init_channels
			init_channels *= growth_per_block
			modules.append(self.conv_unit(prev_channels, init_channels, kernel_size = 3, stride = 2, padding = 1))
			modules.append(self.conv_unit(init_channels, init_channels, kernel_size = 3, stride = 1, padding = 1))
		
		self.base_out_channels = init_channels
		return nn.Sequential(*modules)

	def forward(self, x):
		return self.net(x)

class PoseEncoder(BaseEncoder):
	def __init__(self, K = 10):
		super(PoseEncoder, self).__init__()
		self.num_landmarks = K
		self.conv1x1 = self.conv_unit(self.base_out_channels, self.num_landmarks, kernel_size = 1, stride = 1, padding = 0, batch_norm = False, activation_relu = False)
		# self.sm_row = nn.Softmax(dim = 2)
		# self.sm_col = nn.Softmax(dim = 3)

	def get_coordinates(self, x):

		mean_row = x.mean(dim = 2, keepdim = True)
		mean_col = x.mean(dim = 3, keepdim = True)
		# mean_row = x.mean(dim = 2, keepdim = True).sum(dim = 3)
		# mean_col = x.mean(dim = 3, keepdim = True).sum(dim = 2)
		# print(mean_row, mean_col)
		# print(mean_row.size(), mean_col.size())

		sm_row = nn.Softmax(dim = 3)(mean_row)
		sm_col = nn.Softmax(dim = 2)(mean_col)
		# print(sm_row.size())
		# print(sm_col.size())
		# sm_row = nn.Softmax(dim = 3)(mean_row).sum(dim = 3)
		# sm_col = nn.Softmax(dim = 2)(mean_col).sum(dim = 2)
		# print(sm_row, sm_col)
		# print(sm_row.size(), sm_col.size())

		_, x_coord = sm_row.max(dim = 3, keepdim = True)
		_, y_coord = sm_col.max(dim = 2, keepdim = True)
		# print(x_coord.size())
		# print(y_coord.size())

		# LINSPACE CODE HERE #########

		# # for i in range(self.num_landmarks):
		# x_linspace = torch.linspace(-1.0, 1.0, steps = sm_row.size(3)).repeat(sm_row.size(0), sm_row.size(1), 1).view(sm_row.size(0), sm_row.size(1), 1, -1)
		# y_linspace = torch.linspace(-1.0, 1.0, steps = sm_col.size(2)).repeat(sm_col.size(0), sm_col.size(1), 1).view(sm_col.size(0), sm_col.size(1), -1, 1)
		# # print(x_linspace.size(), y_linspace.size())

		# x_linspace = torch.tensor(x_linspace, requires_grad = False).to(device)
		# y_linspace = torch.tensor(y_linspace, requires_grad = False).to(device)
		
		# b = sm_row.size(0)
		# c = sm_row.size(1)
		# h = sm_col.size(2)
		# w = sm_row.size(3)
		# x_coord = torch.bmm(sm_row.view(b * c, 1, w), x_linspace.view(b * c, w, 1)).view(b, c, 1, 1)
		# y_coord = torch.bmm(sm_col.view(b * c, 1, h), y_linspace.view(b * c, h, 1)).view(b, c, 1, 1)
		# # print(x_coord.size(), y_coord.size())
		# # print(x_linspace.sum(dim = 3), x_linspace.sum(dim = 2), y_linspace.sum(dim = 2), y_linspace.sum(dim = 3))

		# x_coord = (x_coord * (w / 2) + (w / 2)) % w
		# y_coord = (y_coord * (h / 2) + (h / 2)) % h
		# # x_coord = torch.tensor(x_coord, dtype = torch.long).to(device)
		# # y_coord = torch.tensor(y_coord, dtype = torch.long).to(device)
		# x_coord = x_coord.to(dtype = torch.long)
		# y_coord = y_coord.to(dtype = torch.long)
		##############################


		# print(max_row, max_col)
		# print(max_row.size(), max_col.size())

		x_coord = x_coord.view(x_coord.size(0), x_coord.size(1), -1)
		y_coord = y_coord.view(y_coord.size(0), y_coord.size(1), -1)
		# print(x_coord, y_coord)
		# print(x_coord.size(), y_coord.size())
		# return coordinates
		return torch.cat((y_coord, x_coord), 2)

	def get_maps(self, coordinates, map_size = (16, 16)):

		dims = coordinates.size()
		maps = torch.zeros(dims[0], dims[1], map_size[0], map_size[1])

		for i in range(dims[0]):
			for j in range(dims[1]):
				maps[i][j][coordinates[i][j][0]][coordinates[i][j][1]] = 1.0

		return maps

	def get_gaussian_kernel(self, kernel_size=7, sigma=1, channels=10):
		'''
		https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/7
		'''
		# Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
		x_coord = torch.arange(kernel_size)
		x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
		y_grid = x_grid.t()
		xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

		mean = (kernel_size - 1)/2.
		variance = sigma**2.

		# Calculate the 2-dimensional gaussian kernel which is
		# the product of two gaussian distributions for two different
		# variables (in this case called x and y)
		gaussian_kernel = (1./(2.*math.pi*variance)) *\
						torch.exp(
							-torch.sum((xy_grid - mean)**2., dim=-1) /\
							(2*variance)
						)

		# Make sure sum of values in gaussian kernel equals 1.
		gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
		# print('Gau', gaussian_kernel.size())

		# Reshape to 2d depthwise convolutional weight
		gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
		gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
		# print('Gau', gaussian_kernel.size())

		gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
									kernel_size=kernel_size, groups=channels, bias=False, padding = 3)

		gaussian_filter.weight.data = gaussian_kernel
		# print('Gau', gaussian_kernel.size())
		
		gaussian_filter.weight.requires_grad = False
		
		return gaussian_filter


	def forward(self, x):
		x = self.net(x)
		x = self.conv1x1(x)

		coordinates = self.get_coordinates(x)
		# print(coordinates.size())

		maps = self.get_maps(coordinates)
		# print('Gau', maps.size())

		self.gau_filter = self.get_gaussian_kernel(channels = maps.size(1))
		maps = self.gau_filter(maps)
		# print(maps.size(), maps.sum())
		return maps

class Decoder(nn.Module):
	def __init__(self, input_channels = 256 + 10, input_res = 16, init_channels = 256, shrink_per_block = 2, output_channels = 3, output_res = 128):
		super(Decoder, self).__init__()
		self.input_channels = input_channels
		self.input_res = input_res
		self.init_channels = init_channels
		self.shrink_per_block = shrink_per_block
		self.output_channels = output_channels
		self.output_res = output_res
		self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False)
		self.net = self.network(self.input_channels, self.input_res, self.init_channels, self.shrink_per_block, self.output_channels, self.output_res)

	def conv_unit(self, in_ch, out_ch, kernel_size = 3, stride = 1, padding = 0, batch_norm = True, activation_relu = True):
		modules = []
		modules.append(nn.Conv2d(in_channels = in_ch, out_channels = out_ch, kernel_size = kernel_size, stride = stride, padding = padding))
		# Paper uses BN before ReLU
		if batch_norm:
			modules.append(nn.BatchNorm2d(num_features = out_ch))
		if activation_relu:
			modules.append(nn.ReLU())
		return nn.Sequential(*modules)


	def network(self, input_channels, input_res, init_channels, shrink_per_block, output_channels, output_res):
		# Upsamples till final res to have 3 channels with no BN + ReLU for final layer.
		# Preserves at least 32 filters (as said in paper) but is written as 8 in original repo
		modules = []
		prev_channels = input_channels
		# print(prev_channels)
		while input_res <= output_res:
			modules.append(self.conv_unit(prev_channels, init_channels, kernel_size = 3, stride = 1, padding = 1))
			if input_res == output_res:
				modules.append(self.conv_unit(init_channels, output_channels, kernel_size = 3, stride = 1, padding = 1, batch_norm = False, activation_relu = False))
				break
			else:
				modules.append(self.conv_unit(init_channels, init_channels, kernel_size = 3, stride = 1, padding = 1))
				modules.append(self.upsample)
				input_res *= 2
			
			if init_channels > 8:
				prev_channels = init_channels
				init_channels = int(init_channels / shrink_per_block)

		return nn.Sequential(*modules)

	def forward(self, x):
		return self.net(x)


class DecoderConvTranspose(nn.Module):
	def __init__(self, input_channels = 256 + 10, input_res = 16, init_channels = 256, shrink_per_block = 2, output_channels = 3, output_res = 128):
		super(DecoderConvTranspose, self).__init__()
		self.input_channels = input_channels
		self.input_res = input_res
		self.init_channels = init_channels
		self.shrink_per_block = shrink_per_block
		self.output_channels = output_channels
		self.output_res = output_res
		# self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False)
		self.net = self.network(self.input_channels, self.input_res, self.init_channels, self.shrink_per_block, self.output_channels, self.output_res)

	def conv_transpose_unit(self, in_ch, out_ch, kernel_size = 3, stride = 2, padding = 1, batch_norm = True, activation_relu = True):
		modules = []
		modules.append(nn.ConvTranspose2d(in_channels = in_ch, out_channels = out_ch, kernel_size = kernel_size, stride = stride, padding = padding))
		# Paper uses BN before ReLU
		if batch_norm:
			modules.append(nn.BatchNorm2d(num_features = out_ch))
		if activation_relu:
			modules.append(nn.ReLU())
		return nn.Sequential(*modules)

	def conv_unit(self, in_ch, out_ch, kernel_size = 3, stride = 1, padding = 0, batch_norm = True, activation_relu = True):
		modules = []
		modules.append(nn.Conv2d(in_channels = in_ch, out_channels = out_ch, kernel_size = kernel_size, stride = stride, padding = padding))
		# Paper uses BN before ReLU
		if batch_norm:
			modules.append(nn.BatchNorm2d(num_features = out_ch))
		if activation_relu:
			modules.append(nn.ReLU())
		return nn.Sequential(*modules)

	def network(self, input_channels, input_res, init_channels, shrink_per_block, output_channels, output_res):
		modules = []
		prev_channels = input_channels
		# print(prev_channels)
		while input_res <= output_res:
			modules.append(self.conv_unit(prev_channels, init_channels, kernel_size = 3, stride = 1, padding = 1))
			if input_res == output_res:
				modules.append(self.conv_unit(init_channels, output_channels, kernel_size = 3, stride = 1, padding = 1, batch_norm = False, activation_relu = False))
				break
			else:
				modules.append(self.conv_unit(init_channels, init_channels, kernel_size = 3, stride = 1, padding = 1))
				s, f, p = 2, 4, 1
				modules.append(self.conv_transpose_unit(init_channels, init_channels, kernel_size = f, stride = s, padding = p))
				input_res = s * (input_res - 1) + f - 2 * p
			
			if init_channels > 8:
				prev_channels = init_channels
				init_channels = int(init_channels / shrink_per_block)

		return nn.Sequential(*modules)

	# def network(self, input_channels, input_res, init_channels, shrink_per_block, output_channels, output_res):
	# 	# Upsamples till final res to have 3 channels with no BN + ReLU for final layer.
	# 	# Preserves at least 32 filters (as said in paper) but is written as 8 in original repo
	# 	modules = []
	# 	prev_channels = input_channels
	# 	# print(prev_channels)
	# 	while input_res <= output_res:
	# 		if input_res == output_res:
	# 			modules.append(self.conv_transpose_unit(init_channels, output_channels, kernel_size = 3, stride = 1, padding = 1, batch_norm = False, activation_relu = False))
	# 			break
	# 		else:
	# 			s, f, p = 1, 5, 1
	# 			modules.append(self.conv_transpose_unit(prev_channels, init_channels, kernel_size = f, stride = s, padding = p))
	# 			# modules.append(self.conv_transpose_unit(init_channels, init_channels, kernel_size = 3, stride = 2, padding = 1))
	# 			# modules.append(self.upsample)
	# 			input_res = s * (input_res - 1) + f - 2 * p
			
	# 		prev_channels = init_channels
	# 		if init_channels > 8:
	# 			init_channels = int(init_channels / shrink_per_block)


	# 	return nn.Sequential(*modules)

	def forward(self, x):
		return self.net(x)




		
class PerceptualLoss(nn.Module):
	def __init__(self, norm = 'L2'):
		super(PerceptualLoss, self).__init__()
		self.norm = norm
		self.vgg = torchvision.models.vgg16(pretrained = True)
		self.vgg_list = list(list(self.vgg.children())[0].children())
		self.weights = np.array([100.0, 1.0, 1.0, 1.0, 1.0, 1.0])
		# self.weights = np.array([100.0, 1.6, 2.3, 1.8, 2.8, 100.0]) # From paper's code
		self.weights = self.weights / self.weights.sum()

		self.names = ['input', 'conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2']
		self.layer_ids = np.array([4, 9, 14, 21, 28], dtype = 'int') - 1
		# print(self.vgg_list)
	
	def forward(self, x_prime = None, x_prime_hat = None):
		
		# return self.vgg(x_prime)
		wt_index = 0
		current_loss = 0.0
		if self.norm == 'L2':
			loss = nn.MSELoss()
			distance = loss(x_prime, x_prime_hat)
			current_loss += (self.weights[wt_index] * distance)
			wt_index += 1
		else:
			raise NotImplementedError('Norms other than L2 not implemented')

		for index in range(len(self.vgg_list)):
			if index in self.layer_ids:
				current_vgg_list = self.vgg_list[0 : (index + 1)]
				self.vgg = nn.Sequential(*current_vgg_list)
				for param in self.vgg.parameters():
					param.requires_grad = False

				# summary(self.vgg, (3, 128, 128))
				features_x_prime = self.vgg(x_prime)
				features_x_prime_hat = self.vgg(x_prime_hat)

				if self.norm == 'L2':
					loss = nn.MSELoss()
					distance = loss(features_x_prime, features_x_prime_hat)
					current_loss += (self.weights[wt_index] * distance)
					wt_index += 1
				else:
					raise NotImplementedError('Norms other than L2 not implemented')
		
		# current_loss /= len(self.vgg_list)
		return current_loss



# base = BaseEncoder().to(device)
# summary(base, (3, 128, 128))

# pose = PoseEncoder().to(device)
# summary(pose, (3, 128, 128))

# decoder = Decoder(input_channels = 266).to(device)
# summary(decoder, (256 + 10, 16, 16))

# decoder = DecoderConvTranspose(input_channels = 266).to(device)
# summary(decoder, (256 + 10, 16, 16))

# loss = PerceptualLoss().to(device)
