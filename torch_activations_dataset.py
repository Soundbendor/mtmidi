import torch
import os
import polars as pl
from sklearn import preprocessing as SKP
import torch.nn.functional as NF

class STPActivationsData(torch.utils.data.Dataset):
    def __init__(self, csvfile=os.path.join('csv', 'polyrhy_split1.csv'), data_folder='jukebox_acts', set_type='train', device='cpu', num_classes = 8, classdict = None, classification = True, layer=-1):
        self.device = device
        cur_data = pl.scan_csv(csvfile).collect()
        self.data =  cur_data.filter(pl.col('set') == set_type)
        self.classification = classification
        if classification == True:
            self.classdict = classdict
            self.num_classes = num_classes
        self.data_folder = data_folder
        self.layer = layer 
    def __len__(self):
        return self.data['name'].count()

    def __getitem__(self, idx):
        cur_truth = None
        cur_name = self.data['name'][idx]
        if self.classification == True:
            cur_label = self.data['poly'][idx]
            cur_truth = self.classdict[cur_label]
        else:
            cur_truth = self.data['ratio'][idx]
        fpath = os.path.join(self.data_folder, f'{cur_name}-trim.pt')
        cur_arr = None
        if self.layer < 0:
            cur_arr = torch.load(fpath, map_location=torch.device(self.device))
        else:
            cur_arr = torch.load(fpath, map_location=torch.device(self.device))[layer]

        #cur_onehot = NF.one_hot(torch.tensor(cur_lidx),  num_classes = self.num_classes)
        return cur_arr, cur_truth



