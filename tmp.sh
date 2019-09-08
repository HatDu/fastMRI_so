# python tools/cal_sensmap.py /home/amax/SDB/fastmri/multicoil_train_all /home/amax/SDB/fastmri/multicoil_trainsens
# python tools/cal_sensmap.py /home/amax/SDB/fastmri/multicoil_val /home/amax/SDB/fastmri/multicoil_valsens
# CUDA_VISIBLE_DEVICES=0,1 python train.py --cfg configs/baseline_unet.py -acq both -l log/baseline_unet_both_128_10
# CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/baseline_unet.py -acq pd -l log/baseline_unet_pd_128_10_2
# sleep 30m
# CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/baseline_unet.py -acq pdfs -l log/baseline_unet_pdfs_128_10
# train w/o rss
# CUDA_VISIBLE_DEVICES=0,1 python train.py --cfg configs/baseline_unet_wo_rss.py -acq both -l log/baseline_unet_wo_rss
# CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/se_unet.py -acq both -l log/baseline_seunet_wo_rss/
# train dilated unet and se unet
# CUDA_VISIBLE_DEVICES=0,1 python train.py --cfg configs/dilatedunet_rss.py -acq both -l log/dilatedunet_rss/
# CUDA_VISIBLE_DEVICES=0,1 python train.py --cfg configs/unet_random_crop.py -acq both -l log/unet_random_crop/
# CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/seunet_rss.py -acq both -l log/seunet_rss/

# train fusion net
# CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/fusion_unet_residual.py -acq both -l log/fusion_unet_residual/
# CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/fusion_unet_guid.py -acq both -l log/fusion_unet_guid/

# train fusion net v2
# CUDA_VISIBLE_DEVICES=0,1 python train.py --cfg configs/fusion_unet_residualv2_patch_img.py -acq both -l log/fusion_unet_residualv2_patch_img/
# CUDA_VISIBLE_DEVICES=0,1 python train.py --cfg configs/fusion_unet_residualv2_whole_img.py -acq both -l log/fusion_unet_residualv2_whole_img/

## unet stack more and deeper
# CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/unet_stack_more.py -acq both -l log/unet_stack_more/
# CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/unet_stack_deeper.py -acq both -l log/unet_stack_deeper/
# CUDA_VISIBLE_DEVICES=0,1 python train.py --cfg configs/unet_stack_more_deeper.py -acq both -l log/unet_stack_more_deeper/

# # train fusion net v3 and v2
# CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/fusion_unetv2.py -acq both -l log/fusion_unetv2/
# CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/fusion_unetv3.py -acq both -l log/fusion_unetv3/

# train complex net
CUDA_VISIBLE_DEVICES=0,1 python train.py --cfg configs/complex_net.py -acq both -l log/complex_net/

# cfg_file='configs/complex_net.py'
# ckpt='log/complex_net/best_model.pt'
# device='0,1'

# # visualize sensity map
# CUDA_VISIBLE_DEVICES=$device python tools/vis_sensmap.py --cfg $cfg_file -c $ckpt \
#     -i data/multicoil_val/ -o data/infer \
#     -a x4 -acq both 

# # val on 20 x4
# rm -r data/infer/*.h5
# CUDA_VISIBLE_DEVICES=$device python infer.py --cfg $cfg_file -c $ckpt \
#     -i data/multicoil_val/ -o data/infer \
#     -a x4 -acq both 
# python eval.py --target-path data/multicoil_val/ --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPD_FBK
# python eval.py --target-path data/multicoil_val/ --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPDFS_FBK

# # val on 20 x8
# rm -r data/infer/*.h5
# CUDA_VISIBLE_DEVICES=$device python infer.py --cfg $cfg_file -c $ckpt \
#     -i data/multicoil_val/ -o data/infer \
#     -a x8 -acq both 
# python eval.py --target-path data/multicoil_val/ --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPD_FBK
# python eval.py --target-path data/multicoil_val/ --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPDFS_FBK

# val on 199 x4
# rm -r data/infer/*.h5
# CUDA_VISIBLE_DEVICES=$device python infer.py --cfg $cfg_file  -c $ckpt \
#     -i /home/amax/SDB/fastmri/multicoil_val -o data/infer \
#     -a x4 -acq both 
# python eval.py --target-path /home/amax/SDB/fastmri/multicoil_val --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPD_FBK
# python eval.py --target-path /home/amax/SDB/fastmri/multicoil_val --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPDFS_FBK

# val on 199 x8
# CUDA_VISIBLE_DEVICES=$device python infer.py --cfg $cfg_file -c $ckpt \
#     -i /home/amax/SDB/fastmri/multicoil_val -o data/infer \
#     -a x8 -acq both 
# python eval.py --target-path /home/amax/SDB/fastmri/multicoil_val --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPD_FBK
# python eval.py --target-path /home/amax/SDB/fastmri/multicoil_val --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPDFS_FBK