import os
import torch
from tqdm import tqdm
from configs import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w
import numpy as np
from PIL import Image
import json
from camera_utils import LookAtPoseSampler
class SingleImageCoach(BaseCoach):

    def __init__(self,trans):
        super().__init__(data_loader=None, use_wandb=False)
        self.source_transform = trans
    

    # def load_model_from_path(path):


    def train(self, image_path, w_path,c_path):

        use_ball_holder = True

        name = os.path.basename(w_path)[:-4]
        print("image_path: ", image_path, 'c_path', c_path)
        c = np.load(c_path)

        c = np.reshape(c, (1, 25))

        c = torch.FloatTensor(c).cuda()

        from_im = Image.open(image_path).convert('RGB')

        if self.source_transform:
            image = self.source_transform(from_im)
        

        


        print("print 1 = ", name, int(image_path.split("/")[-1].split(".")[0].split("-")[-1]))
        if int(image_path.split("/")[-1].split(".")[0].split("-")[-1]) != 1:
            ckpt = torch.load(f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_align-001_w.pth')
            self.G.load_state_dict(ckpt['G_ema'], strict=False)
        else:
            self.restart_training()



        print('load pre-computed w from ', w_path)
        if not os.path.isfile(w_path):
            print(w_path, 'does not exist!')
            return None

        w_pivot = torch.from_numpy(np.load(w_path)).to(global_config.device)



        
        w_pivot = torch.from_numpy(np.load('../../crop_out/align-001_w/align-001_w.npy')).to(global_config.device)

        # w_pivot = w_pivot.detach().clone().to(global_config.device)
        w_pivot = w_pivot.to(global_config.device)

        log_images_counter = 0
        real_images_batch = image.to(global_config.device)

        camera_lookat_point = torch.tensor([0, 0, 0.2], device=global_config.device) 
        pbar = tqdm(range(hyperparameters.max_pti_steps))
        for i in pbar:


            generated_images = self.forward(w_pivot, c)
            loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, name,
                                                           self.G, use_ball_holder, w_pivot)

            self.optimizer.zero_grad()

            if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                break

            loss.backward()
            self.optimizer.step()

            use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0


            global_config.training_step += 1
            log_images_counter += 1
            
            pbar.set_description("l2: " + str(l2_loss_val) + "  lpips: " +str(loss_lpips))
        
        #Chaneg Camera Params:
        pitch_range = 0.25
        yaw_range = 0.35


        rate = 9/10
        cam2world_pose = LookAtPoseSampler.sample(
            3.14 / 2 + yaw_range * np.sin(2 * 3.14 * rate),
            3.14 / 2 - 0.05 + pitch_range * np.cos(2 * 3.14 * rate),
            camera_lookat_point, radius=2.7, device=global_config.device)
        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=global_config.device)
        c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        generated_images = self.forward(w_pivot, c)

        vis_img = (generated_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        Image.fromarray(vis_img[0].detach().cpu().numpy(), 'RGB').save(f'temp_imgs/{global_config.run_name}_{name}.png')

        self.image_counter += 1

        save_dict = {
            'G_ema': self.G.state_dict()
        }
        # checkpoint_path = f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_{name}.pth'
        checkpoint_path = f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_align-001_w.pth'
        print('final model ckpt save to ', checkpoint_path)
        torch.save(save_dict, checkpoint_path)




        import pickle
        with open('losses.pkl', 'rb') as f:
            stored_losses = pickle.load(f)


        stored_losses[f'{global_config.run_name}_{name}'] = [l2_loss_val, loss_lpips]

        with open('Experiments.pkl', 'wb') as f:
            pickle.dump(stored_losses, f)
