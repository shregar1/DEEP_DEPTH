import h5py
from prettytable import PrettyTable
from data.dataset import Image_Dataset
from torch.utils.data import DataLoader

class Utils:
    @classmethod
    def get_loader(cls,images,masks, image_size,batch_size,num_workers,CUDA):
        dataset = Image_Dataset(images,masks,image_size =image_size,CUDA=CUDA)
        data_loader = DataLoader(dataset=dataset,batch_size=batch_size,
                                 shuffle=True,num_workers=num_workers)
        return data_loader
    
    @classmethod
    def load_data(cls,filename,fields):
        f=h5py.File(filename,"r")
        images=f[fields[0]]
        ground_truth=f[fields[1]]
        return images,ground_truth
    
    @classmethod
    def train_val_split(cls,filename,fields,train_size):
        images,ground_truth=cls.load_data(filename,fields)
        train_images=images[:train_size]
        train_ground_truth=ground_truth[:train_size]
        val_images=images[train_size:]
        val_ground_truth=ground_truth[train_size:]
        return{"train_data":[train_images,train_ground_truth],
               "val_data":[val_images,val_ground_truth]}
    
    @classmethod
    def print_metrics(cls,avg_loss):
        t = PrettyTable(['Parameter', 'Value'])
        t.add_row(['Avg_Loss', avg_loss])
        print(t)
        return
    
    @classmethod
    def shut_down_encoder(cls,model, encoder_layer_num, shut_encoder):
        i=0
        for parameter in model.parameters():
            parameter.requires_grad = shut_encoder
            i+=1
            if(i==encoder_layer_num):
                break
        i=0
        for parameter in model.parameters():
            if parameter.requires_grad :
                print(i,"True",parameter.data.shape)
            else:
                print(i,"False",parameter.shape)
            i+=1
        return 
    