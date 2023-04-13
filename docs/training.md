## Prerequirement

Make sure you have already generated all the required synthetic data (refer to [ICON's Instruction](https://github.com/YuliangXiu/ICON/blob/master/docs/dataset.md)) 
and add a `_1024_'` tobe `./data/thuman2_1024_{num_views}views`, which includes the rendered RGB (`render/`), normal images(`normal_B/`, `normal_F/`, `T_normal_B/`, `T_normal_F/`), corresponding calibration matrix (`calib/`) and pre-computed visibility arrays (`vis/`).


## Command

```bash
# First Train the normal and depth prediction net(or use a pretrained network)

CUDA_VISIBLE_DEVICES=0 python -m apps.train-normal -cfg ./configs/train/normal1024.yaml

CUDA_VISIBLE_DEVICES=0 python -m apps.train-depth -cfg ./configs/train/depth1024.yaml

# Generate Predicted Normal and Depth map
# The Generated Normal map will in 'F_noraml_F' and 'F_normal_B'
# Depth map will in 'F_depth_F' and 'F_depth_B' under  `./data/thuman2_1024_{num_views}views`

# Remember to store your model according to the configuration file
CUDA_VISIBLE_DEVICES=0 python -m apps.train-normal -cfg ./configs/train/normal1024.yaml -test
CUDA_VISIBLE_DEVICES=0 python -m apps.train-depth -cfg ./configs/train/depth1024.yaml -test


# Training for coarse-IF
CUDA_VISIBLE_DEVICES=0 python -m apps.train-coarse -cfg ./configs/train/icon-coarse-nofilter.yaml

# Training for fine-IF
CUDA_VISIBLE_DEVICES=0 python -m apps.train-MR -cfg ./configs/train/mlif.yaml
```

## Checkpoint

All the checkpoints are saved at `./data/ckpt/{name}`