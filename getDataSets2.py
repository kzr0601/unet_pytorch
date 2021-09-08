import numpy as np
import cv2
import os
from copy import deepcopy
import torch
from PIL import Image
from torchvision import transforms

height = 540
width = 640

class MyDataset(torch.utils.data.Dataset):
  def __init__(self, image_path, mask_path, rotation_degree=10, brightness=0.1, contrast=0.1, mode="train", datasize=None):
    self.img_path = image_path
    self.mask_path = mask_path
    files = os.listdir(image_path)
    self.size = len(files)
      
    print(mode, image_path, mask_path, f"datasize: {self.size}")

    # for limit data set size
    self.mode = mode 
    if datasize is not None:
      self.size = min(self.size, datasize)
      self.imagelist = self.imagelist[0:self.size]
      self.masklist = self.masklist[0:self.size]
    
    # for train data aug
    self.rotation_degree = rotation_degree
    self.ColorJitter = transforms.ColorJitter(brightness=brightness, contrast=contrast)

    # for debug
    self.debug_index = 0

  def no_aug_transform(self, img_filename, mask_filename):
    img = Image.fromarray(cv2.imread(img_filename, 0).astype(np.uint8))
    msk = Image.fromarray(cv2.imread(mask_filename, 0).astype(np.uint8))
    transform = transforms.Compose([
  		transforms.Resize([height, width]),
  		transforms.ToTensor()
  	])
    return transform(img), transform(msk)
  
  def aug_transform(self, img_filename, mask_filename):
    img = cv2.imread(img_filename, 0)
    msk = cv2.imread(mask_filename, 0)
    if(self.debug_index < 20):
      cv2.imwrite(f"./debug_aug2/{self.debug_index}.png", img)
      cv2.imwrite(f"./debug_aug2/{self.debug_index}_mask.png", msk)

    trainTransform = transforms.Compose([
      transforms.RandomRotation(self.rotation_degree, expand=True, fill=int(img[0][0])),
      transforms.RandomHorizontalFlip(),
      transforms.Resize([height, width])
      ])

    cmb = np.zeros((height, width, 3))
    cmb[:, :, 0] = img
    cmb[:, :, 1] = cmb[:, :, 2] = msk
    combine = Image.fromarray(cmb.astype(np.uint8))
    combine = trainTransform(combine)

    data = np.array(combine)
    image = data[:, :, 0]
    mask = data[:, :, 1]
    mask[mask<125] = 0

    pil_image = Image.fromarray(image.astype(np.uint8))
    pil_image = self.ColorJitter(pil_image)
    
    image = np.array(pil_image)
    if(self.debug_index < 20):
      cv2.imwrite(f"./debug_aug2/{self.debug_index}_aug.png", image.astype(np.uint8))
      cv2.imwrite(f"./debug_aug2/{self.debug_index}_aug_mask.png", mask)
      self.debug_index += 1

    return transforms.ToTensor()(Image.fromarray(image)), transforms.ToTensor()(Image.fromarray(mask))

  def data_aug(self, idx):
    # filename is solid
    img_filename = os.path.join(self.img_path, str(idx)+".png")
    mask_filename= os.path.join(self.mask_path, str(idx)+"_mask.png")
    
    if self.mode == 'train':
      img, mask = self.aug_transform(img_filename, mask_filename)
    else: #elif self.mode == 'valid':
      img, mask = self.no_aug_transform(img_filename, mask_filename) 
    return img, mask

  def preprocess(self, img, mask):
    img_ndarray = np.asarray(img)
    msk_ndarray = np.asarray(msk)
    img_ndarray = (img_ndarray-np.min(img_ndarray) ) / (np.max(img_ndarray) - np.min(img_ndarray) )
    msk_ndarray = (msk_ndarray-np.min(msk_ndarray) ) / (np.max(msk_ndarray) - np.min(msk_ndarray) )
    return torch.as_tensor(img_ndarray.copy()).float().contiguous(), torch.as_tensor(msk_ndarray.copy()).float().contiguous()

  def __getitem__(self, idx):
    img, msk = self.data_aug(idx)
    return img, msk

  def __len__(self):
    return self.size

