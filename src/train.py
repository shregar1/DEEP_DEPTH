import os
import torch
from utils.utils import Utils
from training.trainer import Trainer
from models.dense_unet import DenseUnet


device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA = torch.cuda.is_available()
root_dir = "dataset"
dataset_file = "nyu_depth_v2_labeled.mat"
dataset_path = os.path.join(root_dir, dataset_file)
output_dir = "outputs"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
weight_dir = "weights"
encoder_file= None
if encoder_file is not None:
    encoder_path = os.path.join(weight_dir, encoder_file)
else:
    encoder_path = None
model_weights_file = None
image_size = 320
batch_size = 5
epochs = 30
lr = 0.00008
best_loss = 100000

shut_encoder = False

d_encoder_layer_num = {
    "densenet121": 121,
    "densenet161": 161,
    "densenet169": 169,
    "densenet201": 201,
    "densenet264": 264
}
encoder_arch = "densenet169"
input_channels = 3

model = DenseUnet(encoder_arch=encoder_arch,
                  input_channels=input_channels,
                  encoder_path=encoder_path)

if model_weights_file is not None:
    model_path = os.path.join(weight_dir, model_weights_file)
    model.load_state_dict(torch.load(model_path))
else:
    pass


Utils.shut_down_encoder(model=model, 
                        encoder_layer_num=d_encoder_layer_num.get(encoder_arch),
                        shut_encoder= not shut_encoder)
    
model = model.to(device)
train=Trainer(device=device, CUDA=CUDA, dataset_path=dataset_path, 
              output_dir=output_dir, image_size=image_size, 
              batch_size=batch_size, model=model, epochs=epochs,
              lr=lr, best_loss=best_loss)