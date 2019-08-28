python tools/cal_sensmap.py /home/amax/SDB/fastmri/multicoil_train_all /home/amax/SDB/fastmri/multicoil_trainsens
python tools/cal_sensmap.py /home/amax/SDB/fastmri/multicoil_val /home/amax/SDB/fastmri/multicoil_valsens
# CUDA_VISIBLE_DEVICES=0,1 python train.py --cfg configs/baseline_unet.py -acq both -l log/baseline_unet_pdfs_128_10
# CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/baseline_unet.py -acq pd -l log/baseline_unet_pdfs_128_10
# sleep 30m
# CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/baseline_unet.py -acq pdfs -l log/baseline_unet_pdfs_128_10

# python infer.py --cfg configs/baseline_unet.py -c log/baseline_unet_pdfs_128_10/best_model.pt \
#     -i data/multicoil_val/ -o data/infer \
#     -a x4 -acq both 
# python eval.py --target-path data/multicoil_val/ --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPD_FBK
# python eval.py --target-path data/multicoil_val/ --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPDFS_FBK
# x4 pd,pdfs
# MSE = 1.113e-10 +/- 1.961e-10 NMSE = 0.01276 +/- 0.006801 PSNR = 33.36 +/- 5.044 SSIM = 0.8656 +/- 0.0713
# MSE = 1.442e-11 +/- 1.827e-11 NMSE = 0.01616 +/- 0.01286 PSNR = 34.43 +/- 4.086 SSIM = 0.8278 +/- 0.1614
# x8
# MSE = 2.769e-10 +/- 4.667e-10 NMSE = 0.03317 +/- 0.01889 PSNR = 29.23 +/- 5.128 SSIM = 0.801 +/- 0.09729
# MSE = 2.357e-11 +/- 2.952e-11 NMSE = 0.02605 +/- 0.01392 PSNR = 32.25 +/- 2.413 SSIM = 0.78 +/- 0.1626

# python infer.py --cfg configs/baseline_unet.py -c log/baseline_unet_pdfs_128_10/best_model.pt \
#     -i /home/amax/SDB/fastmri/multicoil_val -o data/infer \
#     -a x4 -acq both 
# python eval.py --target-path /home/amax/SDB/fastmri/multicoil_val --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPD_FBK
# python eval.py --target-path /home/amax/SDB/fastmri/multicoil_val --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPDFS_FBK

# all x4
# MSE = 1.653e-10 +/- 2.738e-10 NMSE = 0.01424 +/- 0.008529 PSNR = 33.31 +/- 3.581 SSIM = 0.8712 +/- 0.05522
# MSE = 1.461e-11 +/- 2.924e-11 NMSE = 0.01477 +/- 0.01473 PSNR = 35.13 +/- 4.237 SSIM = 0.8499 +/- 0.1242

# x8
# MSE = 4.204e-10 +/- 6.629e-10 NMSE = 0.03696 +/- 0.0203 PSNR = 29.17 +/- 3.459 SSIM = 0.8043 +/- 0.07441
# MSE = 2.544e-11 +/- 4.993e-11 NMSE = 0.02493 +/- 0.01666 PSNR = 32.79 +/- 3.833 SSIM = 0.8069 +/- 0.1333