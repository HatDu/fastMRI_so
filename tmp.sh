# sleep 1h
# CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/baseline_unet.py -acq pd -l log/baseline_unet_pdfs
# sleep 30m
# CUDA_VISIBLE_DEVICES=0,1 python train.py --cfg configs/baseline_unet.py -acq pdfs -l log/baseline_unet_pdfsfs

python infer.py --cfg configs/baseline_unet.py -c log/baseline_unet_pdfs/best_model.pt \
    -i data/multicoil_val/ -o data/infer \
    -a x8 -acq both 
python eval.py --target-path data/multicoil_val/ --predictions-path data/infer \
    --challenge multicoil --acquisition CORPD_FBK
python eval.py --target-path data/multicoil_val/ --predictions-path data/infer \
    --challenge multicoil --acquisition CORPDFS_FBK
# x4 pd,pdfs
# MSE = 5.693e-11 +/- 7.386e-11 NMSE = 0.007545 +/- 0.004661 PSNR = 35.7 +/- 5.079 SSIM = 0.9041 +/- 0.06301
# MSE = 1.375e-11 +/- 1.862e-11 NMSE = 0.0156 +/- 0.01519 PSNR = 34.66 +/- 4.629 SSIM = 0.8303 +/- 0.165
# x8
# MSE = 1.44e-10 +/- 2.167e-10 NMSE = 0.01807 +/- 0.01062 PSNR = 31.88 +/- 4.533 SSIM = 0.8541 +/- 0.0788
# MSE = 2.208e-11 +/- 2.738e-11 NMSE = 0.02442 +/- 0.01221 PSNR = 32.53 +/- 2.657 SSIM = 0.7843 +/- 0.1656

python infer.py --cfg configs/baseline_unet.py -c log/baseline_unet_pdfs/best_model.pt \
    -i /home/amax/SDB/fastmri/multicoil_val -o data/infer \
    -a x8 -acq both 
python eval.py --target-path /home/amax/SDB/fastmri/multicoil_val --predictions-path data/infer \
    --challenge multicoil --acquisition CORPD_FBK
python eval.py --target-path /home/amax/SDB/fastmri/multicoil_val --predictions-path data/infer \
    --challenge multicoil --acquisition CORPDFS_FBK

# all x4
# MSE = 9.026e-11 +/- 1.464e-10 NMSE = 0.00825 +/- 0.005189 PSNR = 35.72 +/- 3.574 SSIM = 0.9092 +/- 0.05276
# MSE = 1.422e-11 +/- 3.185e-11 NMSE = 0.01442 +/- 0.01933 PSNR = 35.33 +/- 4.474 SSIM = 0.8523 +/- 0.1262

# x8
# MSE = 2.225e-10 +/- 3.726e-10 NMSE = 0.01962 +/- 0.01172 PSNR = 31.93 +/- 3.407 SSIM = 0.8593 +/- 0.0676
# MSE = 2.366e-11 +/- 4.807e-11 NMSE = 0.02322 +/- 0.01979 PSNR = 33.13 +/- 4.005 SSIM = 0.8116 +/- 0.1349