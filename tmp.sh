# CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/baseline_unet.py -acq pd -l log/baseline_unet_pd_128_10
# sleep 30m
CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/baseline_unet.py -acq pdfs -l log/baseline_unet_pdfs_128_10

# python infer.py --cfg configs/baseline_unet.py -c log/baseline_unet_pd_128_10/best_model.pt \
#     -i data/multicoil_val/ -o data/infer \
#     -a x8 -acq both 
# python eval.py --target-path data/multicoil_val/ --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPD_FBK
# python eval.py --target-path data/multicoil_val/ --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPDFS_FBK
# x4 pd,pdfs
# MSE = 5.423e-11 +/- 6.699e-11 NMSE = 0.007314 +/- 0.004762 PSNR = 35.86 +/- 5.212 SSIM = 0.9055 +/- 0.06364
# MSE = 1.364e-11 +/- 1.658e-11 NMSE = 0.0155 +/- 0.01243 PSNR = 34.61 +/- 4.233 SSIM = 0.8291 +/- 0.1641
# x8
# MSE = 1.44e-10 +/- 2.167e-10 NMSE = 0.01807 +/- 0.01062 PSNR = 31.88 +/- 4.533 SSIM = 0.8541 +/- 0.0788
# MSE = 2.208e-11 +/- 2.738e-11 NMSE = 0.02442 +/- 0.01221 PSNR = 32.53 +/- 2.657 SSIM = 0.7843 +/- 0.1656

# python infer.py --cfg configs/baseline_unet.py -c log/baseline_unet_pd_128_10/best_model.pt \
#     -i /home/amax/SDB/fastmri/multicoil_val -o data/infer \
#     -a x8 -acq both 
# python eval.py --target-path /home/amax/SDB/fastmri/multicoil_val --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPD_FBK
# python eval.py --target-path /home/amax/SDB/fastmri/multicoil_val --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPDFS_FBK

# all x4
# MSE = 8.787e-11 +/- 1.483e-10 NMSE = 0.00799 +/- 0.005316 PSNR = 35.87 +/- 3.664 SSIM = 0.911 +/- 0.05342
# MSE = 1.513e-11 +/- 4.704e-11 NMSE = 0.01515 +/- 0.03425 PSNR = 35.31 +/- 4.698 SSIM = 0.8514 +/- 0.1269

# x8
# MSE = 2.146e-10 +/- 3.543e-10 NMSE = 0.01909 +/- 0.01201 PSNR = 32.07 +/- 3.448 SSIM = 0.8616 +/- 0.06716
# MSE = 2.498e-11 +/- 5.862e-11 NMSE = 0.02443 +/- 0.03318 PSNR = 33.01 +/- 4.188 SSIM = 0.8097 +/- 0.1353