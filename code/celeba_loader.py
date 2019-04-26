import torch
import torchvision
from torchvision import transforms
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import scipy.spatial.distance as ssd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torchsummary import summary
import matplotlib.pyplot as plt

import tpsloader
# import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from PIL import Image
# import sys
# import os
# sys.path.append(os.path.join(os.getcwd(), 'tpsrepo/'))
# from tpsrepo import thinplate as tps
# import cv2

device = 'cuda:0'
def save_figs(x, x_prime, x_prime_hat, landmark_map, y_coords, x_coords, iter_count):
	try:
		os.mkdir('results')
	except:
		pass

	plt.clf()
	npimg = x.cpu().detach().numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.savefig('results/' + str(iter_count) + '_x.png')

	plt.clf()
	npimg = x_prime_hat.cpu().detach().numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.savefig('results/' + str(iter_count) + '_x_prime_hat.png')

	plt.clf()
	npimg = x_prime.cpu().detach().numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.scatter(x_coords.cpu().detach().numpy(), y_coords.cpu().detach().numpy(), s = 40, c = (np.arange(10)*10))
	plt.savefig('results/' + str(iter_count) + '_x_prime.png')

	plt.clf()
	npimg = landmark_map.cpu().detach().numpy()
	plt.imshow(npimg, cmap = 'gray')
	plt.savefig('results/' + str(iter_count) + '_landmarks.png')






def show(img):
	# print(img.size())
	npimg = img.cpu().detach().numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

def show_single_channel(img):
	print(img.size())
	npimg = img.cpu().detach().numpy()
	plt.imshow(npimg, cmap = 'gray')
	plt.show()

def show_landmarks(img, y_coords, x_coords):
	npimg = img.cpu().detach().numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.scatter(x_coords.cpu().detach().numpy(), y_coords.cpu().detach().numpy(), s = 40, c = (np.arange(10)*10))
	plt.show()

class CelebALoader(Dataset):
	def __init__(self, data_dir = 'data/celeba/Img/img_align_celeba_hq', fraction = 1.0, augmentation = None, mode = 'train'):
		super(CelebALoader, self).__init__()
		self.data_dir = data_dir
		self.fraction = fraction
		self.augmentation = augmentation
		self.mode = mode
		self.abs_data_dir = os.path.join(os.getcwd(), self.data_dir)
		self.image_filenames = os.listdir(self.abs_data_dir)
		self.index_bound = int(self.fraction * len(self.image_filenames))
		# print(self.index_bound)

	def __len__(self):
		return self.index_bound
	
	def __getitem__(self, index):
		if index >= self.index_bound:
			raise IndexError()
		else:
			abs_image_path = os.path.join(self.abs_data_dir, self.image_filenames[index])
			
			# image = tpsloader.custom_tps_function(abs_image_path)
			image = Image.open(abs_image_path)
			# plt.imshow(image)
			# plt.show()
			if self.augmentation:
				image = self.augmentation(image)
			
			# Enter TPS here
			# future_image = image
			# abs_image_path = os.path.join(self.abs_data_dir, self.image_filenames[index+1])
			# future_image = Image.open(abs_image_path)
			# print(image.size())
			
			future_image = Image.open(abs_image_path)
			# plt.imshow(image)
			# plt.show()
			# future_image = tpsloader.custom_tps_function(abs_image_path)
			# plt.imshow(image)
			# plt.show()
			if self.augmentation:
				future_image = self.augmentation(future_image)

			sample = {'x': image, 'x_prime': future_image}
			return sample








from models import BaseEncoder, Decoder, PoseEncoder, PerceptualLoss, DecoderConvTranspose




batch_size = 32
K = 10
num_workers = 8
num_epochs = 100
dset_fraction = 0.05
lr = 1e-3
# cyclic_lr = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
# cyclic_lr_index = 0
# lr = cyclic_lr[cyclic_lr_index]
# reduce_lr_every_n_iterations = 100
lr_reduce_factor = 2
iter_hist_size = 100
loss_reduction_queue_index = 0
loss_reduction_queue = [100.0] * iter_hist_size

weight_decay = 5 * 1e-4
rotation_angle = 20.0
grad_clip = 1.0


base = BaseEncoder().to(device)
# summary(base, (3, 128, 128))

pose = PoseEncoder(K = K).to(device)
# summary(pose, (3, 128, 128))

decoder = Decoder(input_channels = 256 + K).to(device)
# decoder = DecoderConvTranspose(input_channels = 256 + K).to(device)
# summary(decoder, (256 + K, 16, 16))

perceptual_loss = PerceptualLoss().to(device)
# summary(loss, (3, 224, 224))

augmentation = transforms.Compose([transforms.Resize((128, 128)), transforms.RandomRotation(rotation_angle, resample = Image.BICUBIC), transforms.CenterCrop((100, 100)), transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
# augmentation = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
train_data = CelebALoader(data_dir = '/home/kartik/Desktop/celeba/Img/img_align_celeba_hq', fraction = dset_fraction, augmentation = augmentation, mode = 'train')
train_data = DataLoader(train_data, batch_size = batch_size, num_workers = num_workers, shuffle = False)

# loss_fn = nn.MSELoss().to(device)
optimizer = optim.Adam(list(base.parameters()) + list(pose.parameters()) + list(decoder.parameters()), lr = lr, weight_decay = 5.0 * 1e-4)

iter_count = 0
for epoch in range(num_epochs):
	print('Epoch:', epoch)
	current_loss = 0.0
	prev_loss = 0.0
	average_loss = 0.0
	num_batches = len(train_data)
	for index, sample in enumerate(train_data):
		# print(index)
		x = sample['x'].to(device)
		x_prime = sample['x_prime'].to(device)
		# show(x[0])
		# show(x_prime[0])
		encoded_x = base(x)
		landmarks = pose(x_prime).to(device)
		# print(landmarks)
		# print(landmarks.size())
		# show_single_channel(landmarks[0, 0, :, :])
		# print(landmarks)
		# print(torch.ones((16, 16), dtype = torch.int))
		# show(torch.ones((16, 16), dtype = torch.float32))

		renderer_input = torch.cat((encoded_x, landmarks), 1) # channel concat

		x_prime_hat = decoder(renderer_input)
		# show(x_prime_hat[0])

		# plt.imshow(transforms.ToPILImage()(x_prime))
		# plt.imshow(transforms.ToPILImage()(x_prime_hat))
		

		# print(x.size(), x_prime.size(), encoded_x.size(), landmarks.size(), renderer_input.size(), x_prime_hat.size())

		# loss = loss_fn(x_prime_hat, x_prime)


		loss = perceptual_loss(x_prime, x_prime_hat)
		optimizer.zero_grad()
		loss.backward()

		# for p in base.parameters():
		# 	p.grad.data.clamp_(max = grad_clip)
		# for p in pose.parameters():
		# 	p.grad.data.clamp_(max = grad_clip)
		# for p in decoder.parameters():
		# 	p.grad.data.clamp_(max = grad_clip)

		torch.nn.utils.clip_grad_norm_(base.parameters(), grad_clip)
		torch.nn.utils.clip_grad_norm_(pose.parameters(), grad_clip)
		torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

		optimizer.step()
		current_loss += loss.item()
		average_loss += current_loss
		print('\r', index, ':', current_loss, average_loss, end = '')

		# if prev_loss <= current_loss:
		# 	# print('hit')
		# 	loss_reduction_queue[loss_reduction_queue_index] = 0
		# else:
		# 	# print('miss')
		# 	loss_reduction_queue[loss_reduction_queue_index] = 1
		# loss_reduction_queue_index += 1
		# loss_reduction_queue_index = loss_reduction_queue_index % iter_hist_size


		# if sum(loss_reduction_queue) > (0.5 * iter_hist_size):
		# 	pass
		# else:
		# 	lr /= lr_reduce_factor
		# 	optimizer = optim.Adam(list(base.parameters()) + list(pose.parameters()) + list(decoder.parameters()), lr = lr, weight_decay = weight_decay)
		# 	print('lr reduced to', lr)
		# 	loss_reduction_queue = [1] * iter_hist_size

		loss_reduction_queue[loss_reduction_queue_index] = current_loss

		prev_tmp_index = (loss_reduction_queue_index + 1) % iter_hist_size
		if(loss_reduction_queue[prev_tmp_index] < loss_reduction_queue[loss_reduction_queue_index]):
			# cyclic_lr_index += 1
			# cyclic_lr_index %= len(cyclic_lr)
			# lr = cyclic_lr[cyclic_lr_index]
			if(lr < 1e-7):
				lr *= lr_reduce_factor
			else:
				lr = lr / lr_reduce_factor
			optimizer.lr = lr
			loss_reduction_queue = [100.0] * iter_hist_size
			print('changed lr to', lr)


		loss_reduction_queue_index += 1
		loss_reduction_queue_index %= iter_hist_size

		

		prev_loss = current_loss
		current_loss = 0.0
		# print(x_prime.size())



		# if index == (len(train_data) - 1):
		if (iter_count % 100) == 0:
			# show(x[0])
			# show(x_prime[0])
			# show(x_prime_hat[0])
			# for i in range(landmarks.size(1))
			
			
			landmarks_superimposed = landmarks[0, :, :, :].sum(dim = 0, keepdim = True)
			landmarks_superimposed = landmarks_superimposed.view(landmarks_superimposed.size(1), -1)
			# print(landmarks_superimposed.size())
			# show_single_channel(landmarks_superimposed)

			current_landmarks = landmarks[0]
			_, x_coords = current_landmarks.max(dim = 1)[0].max(dim = 1)
			_, y_coords = current_landmarks.max(dim = 2)[0].max(dim = 1)
			y_coords = (torch.tensor(y_coords.double()) / (current_landmarks.size(1) - 1) * (x_prime_hat[0].size(1) - 1))
			x_coords = (torch.tensor(x_coords.double()) / (current_landmarks.size(2) - 1) * (x_prime_hat[0].size(2) - 1))
			y_coords = y_coords.int()
			x_coords = x_coords.int()
			# print(y_coords, x_coords, y_coords.size(), x_coords.size())
			# show_landmarks(x_prime[0], y_coords, x_coords)

			save_figs(x[0], x_prime[0], x_prime_hat[0], landmarks_superimposed, y_coords, x_coords, iter_count)

		iter_count += 1

	average_loss /= num_batches
	name = '/home/kartik/Desktop/basic_celeba_' + str(epoch) + '.pth'
	torch.save({
		'image': base.state_dict(),
		'pose':	pose.state_dict(),
		'decoder': decoder.state_dict(),
		'loss': average_loss,
		'lr': lr
	}, name)


		# if iter_count % reduce_lr_every_n_iterations == 0:
		# 	lr /= lr_reduce_factor
		# 	optimizer = optim.Adam(list(base.parameters()) + list(pose.parameters()) + list(decoder.parameters()), lr = lr, weight_decay = weight_decay)
		# 	print('lr reduced to', lr)