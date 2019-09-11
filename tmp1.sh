# CUDA_VISIBLE_DEVICES=0,1 python train.py --cfg configs/baseline_unet.py -acq both -l log/baseline_unet_pd_128_10
# CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/baseline_unet.py -acq pd -l log/baseline_unet_pd_128_10_2

CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/baseline_unet.py -acq both -l log/baseline_unet_20 --sample_num 20
CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/complex_unet.py -acq both -l log/complex_unet_20 --sample_num 20
CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/baseline_unet.py -acq both -l log/baseline_unet_64 --sample_num 64