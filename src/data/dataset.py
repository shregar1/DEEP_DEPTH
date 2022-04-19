import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class Image_Dataset(Dataset):
    def __init__(self,images,masks,image_size, CUDA):
        self.images=images
        self.masks=masks
        self.image_size=image_size
        self.CUDA=CUDA
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        if self.CUDA:
            FloatTensor=torch.cuda.FloatTensor()
        else:
            FloatTensor=torch.FloatTensor()
        image=np.transpose(self.images[idx],(2,1,0))
        mask=np.floor(np.transpose(self.masks[idx],(1,0)))        
        image=cv2.resize(image,(self.image_size,self.image_size))
        mask=cv2.resize(mask,(self.image_size,self.image_size))
        image = np.transpose(image,(2,0,1))
        X = FloatTensor(image)
        y = FloatTensor(np.expand_dims(mask,axis=0))
        return X,y
