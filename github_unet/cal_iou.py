import cv2
import os
import numpy as np

gt_dir = "../../unet_pretrain/Pytorch-UNet/data/valid_masks"
res_dir = "valid_masks_github"
#res_dir = "../../unet_pretrain/Pytorch-UNet/data/valid_masks_github"
#gt_dir = "debug_masks"
#res_dir = "debug_masks_1"

def iou(img1, img2):
	intesect = img1*img2
	union = img1+img2
	i = np.count_nonzero(intesect)
	u = np.count_nonzero(union)
	return i/u

def main():
	gt_files = os.listdir(gt_dir)
	count = len(gt_files)
	avg = 0
	file_iou = {}
	zero_mask = []

	for file in gt_files:
		gt_filename = os.path.join(gt_dir, file)
		file2 = file[0:file.find('_')]+file[file.find('.'):]
		filename = os.path.join(res_dir, file2)
		gt_img = cv2.imread(gt_filename, 0)
		img = cv2.imread(filename, 0)
		cur_iou = iou(gt_img, img)
		avg += cur_iou
		file_iou[file] = cur_iou
		if(np.count_nonzero(img) == 0):
			zero_mask.append(file) 

	topk = 6
	ious = sorted(file_iou.items(), key = lambda kv:(kv[1], kv[0]))
	mink = ious[:topk]
	maxk = ious[len(ious)-topk:]
	print("topk min iou is: filename | iou")
	for k,v in mink:
		print(k, v)
	print("topk max iou is: filename | iou")
	for k,v in maxk:
		print(k, v)
	print(f"num of zero mask is:{len(zero_mask)}")  

	avg = avg/count
	print(f"avg iou: {avg}")

if __name__ == '__main__':
	main()
