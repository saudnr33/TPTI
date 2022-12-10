
import json
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, required = True, help='folder to store outputs')


args = parser.parse_args()
out_dir = args.out_dir

# 'Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/vid001/epoch_20_000000/dataset.json'


with open('Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/'+out_dir+'/epoch_20_000000/dataset.json', 'r') as openfile:
    file_contents = json.load(openfile)




for name, intrisnic_list in file_contents['labels']:
    intrinsic_matrix = np.array(intrisnic_list)
    np.save(out_dir + "/crop/" + name.split(".")[0], intrinsic_matrix)
