Our work builds on [ICON](https://github.com/YuliangXiu/ICON), thanks to their open source code, please consider citing them

To test for a given image, you can download our pretrained ckpt from [google drive](https://drive.google.com/drive/folders/1FUhOyTG8EB3cVF2aNRRosc_It_0SUqrj?usp=sharing) and put them under `data/ckpt`

then using following command:
```
python -m apps.infer-MR -gpu 0 -in_dir YOUR_IMAGE_PATH -out_dir ./results/  -loop_smpl 100 -loop_cloth 200 -hps_type pymaf
```

if you want to use our repair algorithm
```
python -m apps.infer-MR -gpu 0 -in_dir YOUR_IMAGE_PATH -out_dir ./results/  -loop_smpl 100 -loop_cloth 200 -hps_type pymaf -repair
```
