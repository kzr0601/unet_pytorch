from unet import Unet
from getDataSets2 import MyDataset

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt 
import os
import cv2
import numpy as np
import datetime

from tensorboardX import SummaryWriter

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
height = 540
width = 640
image_size = [height, width]
batch_size = 3
num_epoch = 15
name = "model_9_7_ours"

MASK_THRE = 0.4
contrast = 0.5
brightness = 0.5
rotation_degree = 60

# totensor: converts (H*W*C) to (c*h*w), divide by 255 to range[0.0, 1.0]

def load_train_valid_dataset(train_imgpath, train_maskpath, valid_imgpath, valid_maskpath, batch_size=batch_size, ratio=0.8):

	train_datasets = MyDataset(train_imgpath, train_maskpath, rotation_degree=rotation_degree, brightness=brightness, contrast=contrast, mode="train")
	valid_datasets = MyDataset(valid_imgpath, valid_maskpath, mode="valid" )

	train_iter = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
	valid_iter = torch.utils.data.DataLoader(valid_datasets, batch_size=batch_size, shuffle=True)

	return train_iter, valid_iter

def train(net, train_iter, valid_iter, optimizer, scheduler, loss, modelpath, num_epochs=num_epoch ):

	writer1 = SummaryWriter("runs/exp_9_7_"+name)
	net = net.to(device)

	train_loss, train_acc, train_recall, train_f1, valid_loss, valid_acc, valid_recall, valid_f1 = [], [], [], [], [], [], [], []

	max_acc = 0
	for epoch in range(num_epochs):

		#train
		net.train()
		train_l_sum, train_acc_sum, train_recall_sum, train_num = 0.0, 0.0, 0.0, 0
		debug_index = 0   

		index = 0
		for x, y in train_iter:
			x = x.to(device)
			y = y.to(device)
			#print(x.shape)
			output = net(x)
			
			debug_y = y.cpu().numpy().astype(np.int32)
			debug_output = output.cpu().detach().numpy()

			debug_output[debug_output>=MASK_THRE] = 1
			debug_output[debug_output<MASK_THRE] = 0

			writer1.add_image('input', (x+1).cpu()[0]*127.0, index)
			writer1.add_image('gt', (y+1).cpu()[0]*127.0, index)
			# writer1.add_image('output', debug_output_2[0]*250.0, index)
			# writer1.add_image('output_mask', debug_output[0]*250.0, index)
			writer1.add_image('output', output.cpu()[0]*200.0, index)
			writer1.add_image('output_mask', debug_output[0]*127.0, index)

			index+=1
			# exit()

			debug_sum = debug_y * 10 + debug_output
			debug_acc = np.count_nonzero(debug_sum == 11) / (np.count_nonzero(debug_output == 1)+1e-5)
			debug_recall = np.count_nonzero(debug_sum == 11) / (np.count_nonzero(debug_y == 1)+1e-5)
			#print(np.count_nonzero(debug_y == 0),np.count_nonzero(debug_y == 1))
			#print(width*height*debug_y.shape[0], t1+t2+np.count_nonzero(debug_sum == 10)+np.count_nonzero(debug_sum == 1), t1, t2, np.count_nonzero(debug_sum == 10)+np.count_nonzero(debug_sum == 1))
			# debug
			if(debug_index == 0 and epoch %100 == 0):
				#print("epoch 6: ", debug_acc)
				cv2.imwrite("y.png", debug_y[0][0]*255)
				cv2.imwrite(str(epoch)+".png", debug_output[0][0]*255)
				debug_index += 1

			l = loss(output, y) #.sum()
			optimizer.zero_grad()
			l.backward()
			optimizer.step()

			# calculate accuracy
			train_l_sum += l.item()
			train_acc_sum += debug_acc #(debug_output == debug_y).sum() /(width*height)
			train_recall_sum += debug_recall
			train_num += 1 #y.shape[0]
		scheduler.step()
		#print(train_acc_sum, train_num)

		train_acc_current = train_acc_sum / train_num
		train_recall_current = train_recall_sum / train_num
		f1 = 2 / (1/(train_acc_current+1e-5) + 1/(train_recall_current+1e-5))

		print('epoch %d, loss %.4f, train acc %.3f, train recall %.3f' % (epoch + 1, train_l_sum / train_num, train_acc_current, train_recall_current))
		train_loss.append(train_l_sum / train_num)
		train_acc.append(train_acc_current)
		train_recall.append(train_recall_current)
		train_f1.append(f1)
		
		if(f1 > max_acc):
			max_acc = f1
			torch.save(net, "./bestmodel_"+name+".pkl")
			file = open("record.txt", "w")
			file.write(str(epoch))
			file.close()

		#valid
		if( (epoch+1)%5==0 ):
			valid_loss_sum, valid_acc_sum, valid_recall_sum, valid_num = 0.0, 0.0, 0.0, 0
			with torch.no_grad():
				net.eval()
				for x, y in valid_iter:
					x = x.to(device)
					y = y.to(device)
					#print(np.shape(x.cpu().numpy()))
					output = net(x)
					valid_l = loss(output, y)
					output = output.cpu().numpy()
					output[output>=MASK_THRE] = 1
					output[output<MASK_THRE] = 0
					y = y.cpu().numpy()

					debug_sum = y * 10 + output
					t1, t2 = np.count_nonzero(debug_sum == 11), np.count_nonzero(debug_sum == 0)
					debug_acc = t1 / (np.count_nonzero(output == 1)+1e-5)
					debug_recall = t1 / (np.count_nonzero(y == 1)+1e-5)

					valid_acc_sum += debug_acc
					valid_recall_sum += debug_recall
					valid_loss_sum += valid_l
					valid_num +=1 

				valid_loss_current = valid_loss_sum / valid_num
				valid_acc_current = valid_acc_sum / valid_num
				valid_recall_current = valid_recall_sum / valid_num
				valid_f1_current = 2/(1/(valid_acc_current+1e-5)+1/(valid_recall_current+1e-5))
				print('test loss: %.3f, test acc: %.3f, test recall: %.3f' % (valid_loss_current , valid_acc_current, valid_recall_current))
				valid_acc.append(valid_acc_current)
				valid_loss.append(valid_loss_current)
				valid_recall.append(valid_recall_current)
				valid_f1.append(valid_f1_current)


	print("max_acc: ", max_acc)
	torch.save(net, modelpath)
	return train_loss, train_acc, train_recall, train_f1, valid_loss, valid_acc, valid_recall, valid_f1



def main():
	train_imgpath = "/home/kzr/unet_pretrain/Pytorch-UNet/data/imgs"
	train_maskpath = "/home/kzr/unet_pretrain/Pytorch-UNet/data/masks"
	valid_imgpath = "/home/kzr/unet_pretrain/Pytorch-UNet/data/valid_imgs"
	valid_maskpath = "/home/kzr/unet_pretrain/Pytorch-UNet/data/valid_masks"
	
	modelpath = "./model_"+name+".pkl"
	figurepath = "./info.png"

	train_iter, valid_iter = load_train_valid_dataset(train_imgpath, train_maskpath, valid_imgpath, valid_maskpath, batch_size=batch_size)

	# out_channels represents number of segments desired
	unet = Unet(in_channel=1, out_channel=1)
	#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	#unet = torch.load("./bestmodel_augaug5_60_2.pkl").to(device)
	if(batch_size>=3):
		unet = torch.nn.DataParallel(unet, device_ids=[0,1,2])
	elif(batch_size>1):
		unet = torch.nn.DataParallel(unet, device_ids=[0,1])
	else:
		pass
	#unet = torch.load('./model_aug_50.pkl')

	# loss = torch.nn.CrossEntropyLoss()
	# optimizer = torch.optim.SGD(unet.parameters(), lr=0.01, momentum=0.99)
	loss = torch.nn.BCEWithLogitsLoss()
	optimizer = torch.optim.Adam(unet.parameters(), lr=0.02)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma = 0.1)
	#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones = [10, 90], gamma = 0.5, last_epoch=-1)

	start_time = datetime.datetime.now()
	print(start_time)
	train_loss, train_acc, train_recall, train_f1, valid_loss, valid_acc, valid_recall, valid_f1 = train(unet, train_iter, valid_iter, optimizer, scheduler, loss, modelpath)
	end_time = datetime.datetime.now()
	print(end_time)
	print("time: ", end_time-start_time)

	x = [ _ for _ in range(len(train_loss))]
	axes = plt.figure().subplots(1, 2)
	ax1 = axes[0,0]
	ax2 = axes[0,1]
 
	ax1.plot(x, train_loss, 'b-')
	ax1.plot(x, train_acc, 'r-')
	ax1.plot(x, train_recall, 'g-')
	#ax1.plot(x, train_f1, f1_color)

	x = [_ for _ in range(len(valid_acc)) ]
	ax2.plot(x, valid_loss, 'b-') 
	ax2.plot(x, valid_acc, 'r-')
	ax2.plot(x, valid_recall, 'g-') 
	#ax2.plot(x, valid_f1, f1_color)

	plt.savefig(figurepath)
	plt.show()

	print("end")

if __name__ == '__main__':
	main()