
CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/singlecoil/unet.py -acq both -l log/unet/ --sample_num -1
CUDA_VISIBLE_DEVICES=0,1 python train.py --cfg configs/singlecoil/dncn.py -acq both -l log/dncn/ --sample_num -1