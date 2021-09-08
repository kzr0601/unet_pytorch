import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

class Unet(nn.Module):
	def contracting_block(self, in_channels, out_channels, kernel_size=3):
		block = torch.nn.Sequential(
			torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
			torch.nn.ReLU(),
			torch.nn.BatchNorm2d(out_channels),
			torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
			torch.nn.ReLU(),
			torch.nn.BatchNorm2d(out_channels),
			)
		return block

	def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
		block = torch.nn.Sequential(
			torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
			torch.nn.ReLU(),
			torch.nn.BatchNorm2d(mid_channel),
			torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
			torch.nn.ReLU(),
			torch.nn.BatchNorm2d(mid_channel),
			torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
			)
		return block

	def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
		block = torch.nn.Sequential(
			torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
			torch.nn.ReLU(),
			torch.nn.BatchNorm2d(mid_channel),
			torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
			torch.nn.ReLU(),
			torch.nn.BatchNorm2d(mid_channel),
			torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
			torch.nn.ReLU(),
			torch.nn.BatchNorm2d(out_channels),
			)
		return block

	def __init__(self, in_channel, out_channel):
		super(Unet, self).__init__()
		#encode
		self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=64)
		self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
		self.conv_encode2 = self.contracting_block(64, 128)
		self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
		self.conv_encode3 = self.contracting_block(128, 256)
		self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
		self.conv_encode4 = self.contracting_block(256, 512)
		self.conv_maxpool4 = torch.nn.MaxPool2d(kernel_size=2)
		
		# bottleneck
		# self.bottleneck = torch.nn.Sequential(
		# 	torch.nn.Conv2d(kernel_size=3, in_channels=512, out_channels=1024, padding=1),
		# 	torch.nn.ReLU(),
		# 	torch.nn.BatchNorm2d(1024),
		# 	torch.nn.Conv2d(kernel_size=3, in_channels=1024, out_channels=1024, padding=1),
		# 	torch.nn.ReLU(),
		# 	torch.nn.BatchNorm2d(1024),
		# 	torch.nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
		# 	)
		self.bottleneck = torch.nn.Sequential(
			torch.nn.Conv2d(kernel_size=3, in_channels=256, out_channels=512, padding=1),
			torch.nn.ReLU(),
			torch.nn.BatchNorm2d(512),
			torch.nn.Conv2d(kernel_size=3, in_channels=512, out_channels=512, padding=1),
			torch.nn.ReLU(),
			torch.nn.BatchNorm2d(512),
			torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
			)


		#decode
		self.conv_decode4 = self.expansive_block(1024, 512, 256)
		self.conv_decode3 = self.expansive_block(512, 256, 128)
		self.conv_decode2 = self.expansive_block(256, 128, 64)
		self.final_layer = self.final_block(128, 64, out_channel)

	# [upsampled, bypass(size*2)], crop bypass whose size is doubled, then concat with upsampled
	# def crop_and_concat(self, upsampled, bypass, crop=False):
	# 	if crop:
	# 		h = (bypass.size()[2] - upsampled.size()[2]) // 2
	# 		w = (bypass.size()[3] - upsampled.size()[3]) // 2
	# 	#print(c, bypass.shape[2])
	# 	r_n, d_n = -w, -h
	# 	if(bypass.size()[2] %2 == 1):
	# 		d_n = -h-1
	# 	if(bypass.size()[3] %2 == 1):
	# 		r_n = -w-1
		
	# 	bypass = F.pad(bypass, (-w, r_n, -h, d_n))
	# 	return torch.cat((upsampled, bypass), 1)

	# def crop_and_concat(self, upsampled, bypass, crop=False):
	# 	if crop:
	# 		c = (bypass.size()[2] - upsampled.size()[2]) // 2
		
	# 	bypass = F.pad(bypass, (-c, -c, -c, -c))
	# 	return torch.cat((upsampled, bypass), 1)

	def crop_and_concat(self, upsampled, bypass, crop=False):
		if crop:
			h = (bypass.size()[2] - upsampled.size()[2]) 
			w = (bypass.size()[3] - upsampled.size()[3]) 
		
		upsampled = F.pad(upsampled, (0, w, 0, h))
		return torch.cat((upsampled, bypass), 1)

	def forward(self, x):
		# Encode
		# print(x.shape)
		encode_block1 = self.conv_encode1(x)
		# print(encode_block1.shape)
		encode_pool1 = self.conv_maxpool1(encode_block1)
		# print(encode_pool1.shape)
		encode_block2 = self.conv_encode2(encode_pool1)
		# print(encode_block2.shape)
		encode_pool2 = self.conv_maxpool2(encode_block2)
		# print(encode_pool2.shape)
		encode_block3 = self.conv_encode3(encode_pool2)
		# print(encode_block3.shape)
		encode_pool3 = self.conv_maxpool3(encode_block3)
		# print(encode_pool3.shape)
		#encode_block4 = self.conv_encode4(encode_pool3)
		#encode_pool4 = self.conv_maxpool4(encode_block4)

		# Bottleneck
		bottleneck1 = self.bottleneck(encode_pool3)
		# print(bottleneck1.shape)

		# Decode
		# decode_block4 = self.crop_and_concat(bottleneck1, encode_block4, crop=True)
		# # print(decode_block3.shape)
		# cat_layer3 = self.conv_decode4(decode_block4)
		# decode_block3 = self.crop_and_concat(cat_layer3, encode_block3, crop=True)
		decode_block3 = self.crop_and_concat(bottleneck1, encode_block3, crop=True)
		# print(decode_block2.shape)
		cat_layer2 = self.conv_decode3(decode_block3)
		# print(cat_layer2.shape)
		decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=True)
		# print(decode_block2.shape)
		cat_layer1 = self.conv_decode2(decode_block2)
		# print(cat_layer1.shape)
		decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=True)
		# print(decode_block1.shape)
		final_layer = self.final_layer(decode_block1)
		# print(final_layer.shape)
		out = nn.Sigmoid()(final_layer)
		return out