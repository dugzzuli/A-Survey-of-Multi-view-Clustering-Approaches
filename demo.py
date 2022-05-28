import numpy as np

from utils import process

np.random.seed(0)
import torch

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import argparse
import os
import yaml

if __name__ == '__main__':

    data = '3sources'


    config = yaml.load(open("configMain.yaml", 'r'))

    # input arguments
    parser = argparse.ArgumentParser(description='MVC')
    parser.add_argument('--dataset', nargs='?', default=data)
    parser.add_argument('--View_num', default=config[data]['View_num'])
    parser.add_argument('--norm', default=config[data]['norm'])
    parser.add_argument('--sc', type=float, default=10.0, help='GCN self connection')  # config[data]['sc']
    parser.add_argument('--Weight', nargs='?', default=config['Weight'])
    args, unknown = parser.parse_known_args()

    rownetworks, truefeatures_list, labels, idx_train = process.load_data_mv(args, Unified=False)

    args.rownetworks, args.truefeatures_list, args.labels, args.idx_train = rownetworks, truefeatures_list, labels, idx_train


