# sleep 1h
# CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/baseline_unet.py -acq pd -l log/baseline_unet_pd
# sleep 30m
# CUDA_VISIBLE_DEVICES=0,1 python train.py --cfg configs/baseline_unet.py -acq pdfs -l log/baseline_unet_pdfs

# python infer.py --cfg configs/baseline_unet.py -c log/baseline_unet_pd/best_model.pt \
#     -i data/multicoil_val/ -o data/infer \
#     -a x4 -acq both 
# python eval.py --target-path data/multicoil_val/ --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPD_FBK
# python eval.py --target-path data/multicoil_val/ --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPDFS_FBK
# x4
#pd 5.996e-11 +/- 7.783e-11 NMSE = 0.007918 +/- 0.004785 PSNR = 35.48 +/- 5.028 SSIM = 0.9017 +/- 0.06387
#pdfs 1.402e-11 +/- 1.846e-11 NMSE = 0.01584 +/- 0.0144 PSNR = 34.56 +/- 4.462 SSIM = 0.8291 +/- 0.1643
# x8
# pd MSE = 1.482e-10 +/- 2.173e-10 NMSE = 0.01871 +/- 0.01067 PSNR = 31.71 +/- 4.683 SSIM = 0.8506 +/- 0.07987
# pdfs MSE = 2.258e-11 +/- 2.749e-11 NMSE = 0.02501 +/- 0.01244 PSNR = 32.42 +/- 2.583 SSIM = 0.7823 +/- 0.1645

python infer.py --cfg configs/baseline_unet.py -c log/baseline_unet_pd/best_model.pt \
    -i /home/amax/SDB/fastmri/multicoil_val -o data/infer \
    -a x4 -acq both 
python eval.py --target-path /home/amax/SDB/fastmri/multicoil_val --predictions-path data/infer \
    --challenge multicoil --acquisition CORPD_FBK
python eval.py --target-path /home/amax/SDB/fastmri/multicoil_val --predictions-path data/infer \
    --challenge multicoil --acquisition CORPDFS_FBK

# all x4
#PD 9.499e-11 +/- 1.592e-10 NMSE = 0.008633 +/- 0.005819 PSNR = 35.53 +/- 3.631 SSIM = 0.9071 +/- 0.05413
#PDFS 1.578e-11 +/- 5.663e-11 NMSE = 0.01576 +/- 0.04351 PSNR = 35.25 +/- 4.792 SSIM = 0.8512 +/- 0.1273

# x8
#PD 2.332e-10 +/- 3.882e-10 NMSE = 0.02057 +/- 0.01213 PSNR = 31.73 +/- 3.391 SSIM = 0.856 +/- 0.06825
#PDFS 2.597e-11 +/- 7.06e-11 NMSE = 0.0252 +/- 0.04631 PSNR = 32.98 +/- 4.334 SSIM = 0.8097 +/- 0.1356