# sleep 1h
# CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/baseline_unet.py -acq pd -l log/baseline_unet_pd
# sleep 30m
# CUDA_VISIBLE_DEVICES=0,1 python train.py --cfg configs/baseline_unet.py -acq pdfs -l log/baseline_unet_pdfs

python infer.py --cfg configs/baseline_unet.py -c log/baseline_unet/best_model.pt \
    -i data/multicoil_val/ -o data/infer \
    -a x8 -acq both 
python eval.py --target-path data/multicoil_val/ --predictions-path data/infer \
    --challenge multicoil --acquisition CORPD_FBK
python eval.py --target-path data/multicoil_val/ --predictions-path data/infer \
    --challenge multicoil --acquisition CORPDFS_FBK
# x4
#pd MSE = 6.711e-11 +/- 9.002e-11 NMSE = 0.008778 +/- 0.005333 PSNR = 35.04 +/- 5.214 SSIM = 0.8971 +/- 0.0662
#pdfs MSE = 1.386e-11 +/- 1.607e-11 NMSE = 0.01572 +/- 0.01104 PSNR = 34.5 +/- 3.98 SSIM = 0.8271 +/- 0.1629
# x8
# pd MSE = 1.698e-10 +/- 2.474e-10 NMSE = 0.02163 +/- 0.0131 PSNR = 31.1 +/- 4.885 SSIM = 0.8418 +/- 0.08396
# pdfs MSE = 2.327e-11 +/- 2.752e-11 NMSE = 0.02592 +/- 0.01212 PSNR = 32.25 +/- 2.31 SSIM = 0.779 +/- 0.1632


# python infer.py --cfg configs/baseline_unet.py -c log/baseline_unet/best_model.pt \
#     -i /home/amax/SDB/fastmri/multicoil_val -o data/infer \
#     -a x8 -acq both 
# python eval.py --target-path /home/amax/SDB/fastmri/multicoil_val --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPD_FBK
# python eval.py --target-path /home/amax/SDB/fastmri/multicoil_val --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPDFS_FBK

# all x4
#PD MSE = 1.058e-10 +/- 1.739e-10 NMSE = 0.00957 +/- 0.005942 PSNR = 35.07 +/- 3.595 SSIM = 0.9024 +/- 0.05506
#PDFS MSE = 1.617e-11 +/- 5.208e-11 NMSE = 0.01608 +/- 0.03861 PSNR = 35.07 +/- 4.654 SSIM = 0.8488 +/- 0.1268

# x8
#PD MSE = 2.661e-10 +/- 4.291e-10 NMSE = 0.0237 +/- 0.01403 PSNR = 31.11 +/- 3.426 SSIM = 0.8468 +/- 0.06922
#PDFS MSE = 2.78e-11 +/- 7.601e-11 NMSE = 0.02702 +/- 0.05031 PSNR = 32.68 +/- 4.332 SSIM = 0.8054 +/- 0.1353