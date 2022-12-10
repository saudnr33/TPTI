from random import choice
from string import ascii_uppercase
from torchvision.transforms import transforms
import os
from configs import global_config, paths_config
import glob

from training.coaches.single_image_coach import SingleImageCoach
import argparse

def run_PTI(run_name='', in_dir = "", use_wandb=False, use_multi_id_training=False):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    if run_name == '':
        global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    else:
        global_config.run_name = run_name


    global_config.pivotal_training_steps = 1
    global_config.training_step = 1

    # embedding_dir_path = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}/{paths_config.pti_results_keyword}'
    # os.makedirs(embedding_dir_path, exist_ok=True)

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    coach = SingleImageCoach(trans)

    latent_space = 'w_plus'
    for image_path in glob.glob('../../'+in_dir+'/*.png'):
        name = os.path.basename(image_path)[:-4]
        w_path = '../../'+in_dir+'_out'+ f'/{name}_{latent_space}/{name}_{latent_space}.npy'
        c_path = '../../'+in_dir+ f'/{name}.npy'
        if len(glob.glob(f'./checkpoints/*_{name}_{latent_space}.pth'))>0:
            continue

        if not os.path.exists(w_path):
            continue
        coach.train(image_path = image_path, w_path=w_path,c_path = c_path)

    latent_space = 'w'
    for image_path in glob.glob('../../'+in_dir+'/*.png'):
        name = os.path.basename(image_path)[:-4]
        w_path = '../../'+in_dir+'_out'+f'/{name}_{latent_space}/{name}_{latent_space}.npy'
        c_path = '../../'+in_dir+f'/{name}.npy'
        if len(glob.glob(f'./checkpoints/*_{run_name}_{name}_{latent_space}.pth')) > 0:
            continue

        if not os.path.exists(w_path):
            continue
        coach.train(image_path=image_path, w_path=w_path, c_path=c_path)

    return global_config.run_name


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, required = True, help='folder to store outputs')
    parser.add_argument('--run_name', type=str, required = False, help='folder to store outputs',default= '' )

    args = parser.parse_args()
    in_dir = args.in_dir
    run_name = args.run_name

    run_PTI(run_name=run_name,in_dir = in_dir, use_wandb=False, use_multi_id_training=False)
