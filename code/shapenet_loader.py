import tensorflow as tf
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import torch.nn.functional as F
import torch.optim as optim
import random
# from torchsummary import summary

from models import BaseEncoder, Decoder, PoseEncoder, PerceptualLoss, DecoderConvTranspose

saved_model_name = '/home/kartik/Desktop/transpose_cars_'
# saved_model_name = '/home/cap6412.student4/pa1/chairs_models/transpose_chair_'
abs_data_dir = '/home/kartik/Desktop/cars_with_keypoints'
# abs_data_dir = '/home/cap6412.student4/pa1/chairs_with_keypoints'
abs_target_image_dir = '/home/kartik/Desktop/cars_images'
# abs_target_image_dir = '/home/cap6412.student4/pa1/chairs_images'
results_directory = 'cars_results'
device = 'cuda:0'
SAVE_MODEL_EVERY_N_EPOCH = 100
SAVE_FIG_EVERY_N_STEPS = 200
batch_size = 32
K = 5
num_workers = 8
num_epochs = 500
dset_fraction = 1.0
lr = 1e-3
lr_reduce_factor = 2
iter_hist_size = 100
loss_reduction_queue_index = 0
loss_reduction_queue = [100.0] * iter_hist_size
weight_decay = 5 * 1e-4
rotation_angle = 20.0
grad_clip = 1.0


base = BaseEncoder().to(device)
pose = PoseEncoder(K = K).to(device)
decoder = DecoderConvTranspose(input_channels = 256 + K).to(device)
perceptual_loss = PerceptualLoss().to(device)

# transformation = transforms.Compose([transforms.Resize((128, 128)), transforms.RandomRotation(rotation_angle, resample = Image.BICUBIC), transforms.CenterCrop((100, 100)), transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
transformation = transforms.Compose([transforms.Resize((128, 128)), transforms.CenterCrop((100, 100)), transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])

def save_figs(x, x_prime, x_prime_hat, landmark_map, y_coords, x_coords, iter_count, K):
	try:
		os.mkdir(results_directory)
	except:
		pass

	plt.clf()
	npimg = x.cpu().detach().numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.savefig(results_directory + '/' + str(iter_count) + '_x.png')

	plt.clf()
	npimg = x_prime_hat.cpu().detach().numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.savefig(results_directory + '/' + str(iter_count) + '_x_prime_hat.png')

	plt.clf()
	npimg = x_prime.cpu().detach().numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.scatter(x_coords.cpu().detach().numpy(), y_coords.cpu().detach().numpy(), s = 40, c = (np.arange(K)*10))
	plt.savefig(results_directory + '/' + str(iter_count) + '_x_prime.png')

	plt.clf()
	npimg = landmark_map.cpu().detach().numpy()
	plt.imshow(npimg, cmap = 'gray')
	plt.savefig(results_directory + '/' + str(iter_count) + '_landmarks.png')


class ShapeNetLoader(Dataset):
	def __init__(self, data_dir = 'data/cars_with_keypoints', target_image_folder_name = 'data/cars_images', fraction = 1.0, transformation = None, mode = 'train'):
		self.data_dir = data_dir
		self.fraction = fraction
		self.transformation = transformation
		self.mode = mode
		self.image_dir = target_image_folder_name
		self.index_bound = self.create_image_ds(split = 'train', target_image_folder_name = self.image_dir)
		self.index_bound = int(self.fraction * self.index_bound)
		# for _ in self.dataset:
		# 	self.index_bound += 1
		# self.index_bound *= self.fraction
		print(self.index_bound)


	# Modified to work with tf2.0
	def create_image_ds(self, split, target_image_folder_name):
		"""Returns input_fn for tf.estimator.Estimator.
		Reads tfrecords and construts input_fn for either training or eval. All
		tfrecords not in test.txt or dev.txt will be assigned to training set.
		Args:
		split: A string indicating the split. Can be either 'train' or 'validation'.
		Returns:
		input_fn for tf.estimator.Estimator.
		Raises:
		IOError: If test.txt or dev.txt are not found.
		"""


		vh = vw = 128

		if (not os.path.exists(os.path.join(self.data_dir, "test.txt")) or
		not os.path.exists(os.path.join(self.data_dir, "dev.txt"))):
			raise IOError("test.txt or dev.txt not found")

		with open(os.path.join(self.data_dir, "test.txt"), "r") as f:
			testset = [x.strip() for x in f.readlines()]

		with open(os.path.join(self.data_dir, "dev.txt"), "r") as f:
			validset = [x.strip() for x in f.readlines()]

		files = os.listdir(self.data_dir)
		filenames = []
		for f in files:
			sp = os.path.splitext(f)
			if sp[1] != ".tfrecord" or sp[0] in testset:
				continue

			if ((split == "validation" and sp[0] in validset) or
			(split == "train" and sp[0] not in validset)):
				filenames.append(os.path.join(self.data_dir, f))


		def parser(serialized_example):
			"""Parses a single tf.Example into image and label tensors."""
			# fs = tf.parse_single_example(
			fs = tf.io.parse_single_example(
			serialized_example,
			features={
				"img0": tf.io.FixedLenFeature([], tf.string),
				"img1": tf.io.FixedLenFeature([], tf.string),
				"mv0": tf.io.FixedLenFeature([16], tf.float32),
				"mvi0": tf.io.FixedLenFeature([16], tf.float32),
				"mv1": tf.io.FixedLenFeature([16], tf.float32),
				"mvi1": tf.io.FixedLenFeature([16], tf.float32),
			})

			fs["img0"] = tf.math.divide(tf.cast(tf.image.decode_png(fs["img0"], 4), tf.float32), 255)
			fs["img1"] = tf.math.divide(tf.cast(tf.image.decode_png(fs["img1"], 4), tf.float32), 255)

			fs["img0"].set_shape([vh, vw, 4])
			fs["img1"].set_shape([vh, vw, 4])

			fs["lr0"] = tf.convert_to_tensor([fs["mv0"][0]])
			fs["lr1"] = tf.convert_to_tensor([fs["mv1"][0]])

			return fs

		# np.random.shuffle(filenames)
		dataset = tf.data.TFRecordDataset(filenames, buffer_size = 256)
		dataset = dataset.map(parser)
		
		try:
			os.mkdir(target_image_folder_name)
		except:
			pass # dir already exists
		
		if not os.listdir(target_image_folder_name):
			counter = 0
			for i, elem in enumerate(dataset):
				if (i%100) != 0: # two images per car; since 200 views per car
					continue
				# Ignoring alpha channel
				x = elem['img0'].numpy()[:, :, 0:3]
				img_path = os.path.join(target_image_folder_name, 'x_' + str(int(i/100)) + '.png')
				plt.imsave(img_path, x)
				xprime = elem['img1'].numpy()[:, :, 0:3]
				img_path = os.path.join(target_image_folder_name, 'xprime_' + str(int(i/100)) + '.png')
				plt.imsave(img_path, xprime)

				counter += 1
		else: # use images from previously extracted run
			print('Found files in', self.image_dir)
			counter = int(len(os.listdir(target_image_folder_name)) / 2)
		

		return counter
		
	def __len__(self):
		return self.index_bound
	
	def __getitem__(self, index):
		if index >= self.index_bound:
			raise IndexError()
		else:
			# PIL images
			fname = 'x_' + str(index) + '.png'
			self.abs_filename = os.path.join(self.image_dir, fname)
			self.x = Image.open(self.abs_filename)
			fname = 'xprime_' + str(index) + '.png'

			self.abs_filename = os.path.join(self.image_dir, fname)
			self.x_prime = Image.open(self.abs_filename)
			# plt.imshow(self.x)
			# plt.show()
			# plt.imshow(self.x_prime)
			# plt.show()

			# self.x = Image.fromarray(self.x.astype('uint8'), 'RGB')
			# self.x_prime = Image.fromarray(self.x_prime.astype('uint8'), 'RGB')

			if self.transformation:
				self.x = self.transformation(self.x)
				self.x_prime = self.transformation(self.x_prime)

			sample = {'x': self.x, 'x_prime': self.x_prime}
			return sample



train_data = ShapeNetLoader(data_dir = abs_data_dir, target_image_folder_name = abs_target_image_dir, fraction = dset_fraction, transformation = transformation, mode = 'train')
train_data = DataLoader(train_data, batch_size = batch_size, num_workers = num_workers, shuffle = False)
# print(len(train_data))

optimizer = optim.Adam(list(base.parameters()) + list(pose.parameters()) + list(decoder.parameters()), lr = lr, weight_decay = weight_decay)

# model = torch.load('/home/kartik/Desktop/results_cars_transpose/transpose_car_' + str(99) + '.pth')
# base.load_state_dict(model['image'])
# pose.load_state_dict(model['pose'])
# decoder.load_state_dict(model['decoder'])

# iter_count = 18500
iter_count = 0
# for epoch in range(100, num_epochs):
for epoch in range(0, num_epochs):
	print('Epoch:', epoch)
	current_loss = 0.0
	prev_loss = 0.0
	average_loss = 0.0
	num_batches = len(train_data)
	for index, sample in enumerate(train_data):
		# print(index)
		# for some reason, 4th channel wasn't dropped previously
		x = sample['x'][:, 0:3, :, :].to(device)
		x_prime = sample['x_prime'][:, 0:3, :, :].to(device)
		encoded_x = base(x)
		landmarks = pose(x_prime).to(device)
		renderer_input = torch.cat((encoded_x, landmarks), 1) # channel concat
		x_prime_hat = decoder(renderer_input)

		loss = perceptual_loss(x_prime, x_prime_hat)
		optimizer.zero_grad()
		loss.backward()

		torch.nn.utils.clip_grad_norm_(base.parameters(), grad_clip)
		torch.nn.utils.clip_grad_norm_(pose.parameters(), grad_clip)
		torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

		optimizer.step()
		current_loss += loss.item()
		average_loss += current_loss
		print('\r', index, ':', current_loss, average_loss, end = '')

		loss_reduction_queue[loss_reduction_queue_index] = current_loss

		prev_tmp_index = (loss_reduction_queue_index + 1) % iter_hist_size
		if(loss_reduction_queue[prev_tmp_index] < loss_reduction_queue[loss_reduction_queue_index]):
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

		# if index == (len(train_data) - 1):
		if (iter_count % SAVE_FIG_EVERY_N_STEPS) == 0:			
			landmarks_superimposed = landmarks[0, :, :, :].sum(dim = 0, keepdim = True)
			landmarks_superimposed = landmarks_superimposed.view(landmarks_superimposed.size(1), -1)
			
			current_landmarks = landmarks[0]
			_, x_coords = current_landmarks.max(dim = 1)[0].max(dim = 1)
			_, y_coords = current_landmarks.max(dim = 2)[0].max(dim = 1)
			y_coords = (torch.tensor(y_coords.double()) / (current_landmarks.size(1) - 1) * (x_prime_hat[0].size(1) - 1))
			x_coords = (torch.tensor(x_coords.double()) / (current_landmarks.size(2) - 1) * (x_prime_hat[0].size(2) - 1))
			y_coords = y_coords.int()
			x_coords = x_coords.int()
			
			save_figs(x[0], x_prime[0], x_prime_hat[0], landmarks_superimposed, y_coords, x_coords, iter_count, K)

		iter_count += 1

	average_loss /= num_batches

	if (epoch + 1) % SAVE_MODEL_EVERY_N_EPOCH == 0:
		name = saved_model_name + str(epoch) + '.pth'
		torch.save({
			'image': base.state_dict(),
			'pose':	pose.state_dict(),
			'decoder': decoder.state_dict(),
			'loss': average_loss,
			'lr': lr
		}, name)