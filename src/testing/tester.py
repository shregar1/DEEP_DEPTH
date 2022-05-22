import os
import cv2
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

class Tester:
    def __init__(self, device, CUDA, root_dir, output_dir, image_size, model):
        self.device = device
        self.CUDA = CUDA
        self.root_dir = root_dir
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.image_size=image_size
        self.model = model.to(self.device)
        
    def fit(self):
        for file in tqdm.tqdm(os.listdir(self.root_dir)):
            filepath = os.path.join(self.root_dir, file)
            image = cv2.imread(filename=filepath)
            image=cv2.resize(image,(self.image_size,self.image_size))
            img = np.transpose(image,(2,0,1))
            out = self.model.forward(torch.FloatTensor(img).unsqueeze(0).to(self.device))
            out = np.transpose(np.array(out.detach().cpu())[0],(1,2,0))[:,:,0]
            outpath = os.path.join(self.output_dir, file)
            plt.imsave(outpath, out)
        return