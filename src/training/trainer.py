import os
import tqdm
import torch
import numpy as np
from utils.utils import Utils
from losses.loss import Loss

class Trainer:
    def __init__(self, device, CUDA, dataset_path, output_dir, image_size, batch_size, model, epochs, lr, best_loss):
        self.device = device
        self.CUDA = CUDA
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.out_weight_dir = os.path.join(self.output_dir, "weights")
        if not os.path.exists(self.out_weight_dir):
            os.mkdir(self.out_weight_dir)
        self.image_size=image_size
        self.batch_size=batch_size
        self.model = model.to(self.device)
        self.epochs=epochs
        self.lr=lr
        self.best_loss=best_loss
        
        self.data = Utils.train_val_split(filename=self.dataset_path,
                                          fields=["images","depths"],
                                          train_size=1200)
        self.train_dataset = self.data["train_data"]
        self.valid_dataset = self.data["val_data"]
        self.train_dataloader=Utils.get_loader(images=self.train_dataset[0],
                                               masks=self.train_dataset[1],
                                               image_size=self.image_size,
                                               batch_size=self.batch_size,
                                               num_workers=0,CUDA=self.CUDA)
        self.valid_dataloader=Utils.get_loader(images=self.valid_dataset[0],
                                               masks=self.valid_dataset[1],
                                               image_size=self.image_size,
                                               batch_size=self.batch_size,
                                               num_workers=0,CUDA=self.CUDA)
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        
    def fit(self):
        for epoch in tqdm.tqdm(range(self.epochs)):
            #Training Phase
            self.model.train()
            train_loss = []
            for i, (images, masks) in enumerate(self.train_dataloader):
                images=images/255
                images.to(self.device)
                masks.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = Loss.total_loss(y_pred=outputs, y_true=masks)
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()
  
            #Validation Phase
            self.model.eval()
            valid_loss = []
            for i, (images, masks) in enumerate(self.valid_dataloader):
                images=images/255
                images.to(self.device)
                masks.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = Loss.total_loss(y_pred=outputs, y_true=masks)
                valid_loss.append(loss.item())
            
            avg_train_loss = np.average(train_loss)
            Utils.print_metrics(avg_train_loss)
            
            avg_valid_loss = np.average(valid_loss)
            Utils.print_metrics(avg_valid_loss)
            
            torch.save(self.model.state_dict(),
                       os.path.join(self.out_weight_dir, "Dense_Depth_model.pth"))
            if(avg_valid_loss<self.best_loss):
                torch.save(self.model.state_dict(), 
                           os.path.join(self.out_weight_dir,f"Dense_Depth_model_loss_{avg_valid_loss}.pth"))
                self.best_loss = avg_valid_loss
        return
    