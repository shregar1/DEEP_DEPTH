import os
import torch
from testing.tester import Tester
from models.dense_unet import DenseUnet


device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA = torch.cuda.is_available()
root_dir = "dataset/test"
output_dir = "outputs/results"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
weight_dir = "weights"
model_weights_file = None
image_size = 320

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
                  input_channels=input_channels)

if model_weights_file is not None:
    model_path = os.path.join(weight_dir, model_weights_file)
    model.load_state_dict(torch.load(model_path))
else:
    pass

model = model.to(device)
train=Tester(device=device, CUDA=CUDA, root_dir=root_dir,
             output_dir=output_dir, image_size=image_size, 
             model=model)