# CUDA_VISIBLE_DEVICES=0,1 python train.py --cfg configs/baseline_unet.py -acq both -l log/baseline_unet_pd_128_10
# CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/baseline_unet.py -acq pd -l log/baseline_unet_pd_128_10_2

# 3348227
# CUDA_VISIBLE_DEVICES=0,1 python train.py --cfg configs/baseline_unet.py -acq both -l log/baseline_unet_20 --sample_num 20
# CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/baseline_unet.py -acq both -l log/baseline_unet_64 --sample_num 64
# CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/baseline_unet.py -acq both -l log/baseline_unet_128 --sample_num -1
CUDA_VISIBLE_DEVICES=0,1 python train.py --cfg configs/complex_net.py -acq both -l log/complex_net_20 --sample_num 20
# CUDA_VISIBLE_DEVICES=0,1 python train.py --cfg configs/complex_unet.py -acq both -l log/complex_unet_20 --sample_num 20


# 143055
# CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/cascadenet.py -acq both -l log/cascadenet_20 --sample_num 20