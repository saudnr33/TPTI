# TPTI: Temporal Pivotal Tuning for Latent-based editing of Real Images.



This repository is built on, 


1. **[EG3D](https://github.com/NVlabs/eg3d)**.
2. **[PTI](https://github.com/danielroich/PTI)**.
3. **[ Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/6ba3d22f84bf508f0dde002da8fff277196fef21)**.

Please, follow the intstructions carefully when installing **[EG3D](https://github.com/NVlabs/eg3d)**. 

## Preparing datasets


Run the following commands:
```.bash
cd dataset_preprocessing/ffhq
python preprocess_in_the_wild.py --indir=INPUT_IMAGE_FOLDER
```


## pretrained model

The [projector needs vgg16](https://github.com/oneThousand1000/EG3D-projector/blob/68e44af799b103c75978b11fa825ff9062297c6c/eg3d/projector/w_plus_projector.py#L99) for loss computation, you can download vgg16.pt from https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt and save it to `eg3d/network`.

# Projection:
Run the the following for first frame projection:
```
cd eg3d
python run_projector.py --outdir=projector_out --latent_space_type w  --network=networks/ffhq512-128.pkl --sample_mult=2  --image_path ./projector_test_data/frame001.png --c_path ./projector_test_data/frame001.npy
```


## PTI projector

**Notice:** before you run the PTI, please run the w to get the latent code of the first frame.

Then run:

```
cd eg3d/projector/PTI
python run_pti_video.py --in_dir PATH/TO/DIRECTORY --run_name Experiment_name
```
You can generate a video of the results using:
```
python saud_gen_video.py --in_dir PATH/TO/DIRECTORY
```

