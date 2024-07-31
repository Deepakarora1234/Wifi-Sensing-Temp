from dataset import *
from UT_HAR_model import *

import torch

def load_data_n_model(dataset_name, model_name, root):
    classes = {'UT_HAR_data':7,'NTU-Fi-HumanID':14,'NTU-Fi_HAR':6,'Widar':22}
    if dataset_name == 'UT_HAR_data':
        print('using dataset: UT-HAR DATA')
        data = UT_HAR_dataset(root)
        train_set = torch.utils.data.TensorDataset(data['X_train'],data['y_train'])
        test_set = torch.utils.data.TensorDataset(torch.cat((data['X_val'],data['X_test']),0),torch.cat((data['y_val'],data['y_test']),0))
        train_loader = torch.utils.data.DataLoader(train_set,batch_size=64,shuffle=True, drop_last=True) # drop_last=True
        test_loader = torch.utils.data.DataLoader(test_set,batch_size=256,shuffle=False)
       
        if model_name == 'VGG16':
            print("using model: VGG16")
            model = UT_HAR_VGG16()
            train_epoch = 50 #20
          
        return train_loader, test_loader, model, train_epoch
    
    
    