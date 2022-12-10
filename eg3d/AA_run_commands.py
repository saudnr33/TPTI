

import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--indir', type=str, required=True)
args = parser.parse_args()



#w space projection


sorted_dir = os.listdir(args.indir)

for i in range(len(sorted_dir)):
    if sorted_dir[i].endswith(".png") or sorted_dir[i].endswith(".jpg"):
        name, _ = sorted_dir[i].split(".")

        command = "python run_projector.py --outdir=" + args.indir + "_out" + " --latent_space_type w  --network=networks/ffhq512-128.pkl --sample_mult=2  --image_path ./"+  args.indir +"/" + sorted_dir[i] + " --c_path "+ args.indir + "/" + name +".npy"
        os.system(command)

