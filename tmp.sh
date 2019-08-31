# python tools/cal_sensmap.py /home/amax/SDB/fastmri/multicoil_train_all /home/amax/SDB/fastmri/multicoil_trainsens
# python tools/cal_sensmap.py /home/amax/SDB/fastmri/multicoil_val /home/amax/SDB/fastmri/multicoil_valsens
# CUDA_VISIBLE_DEVICES=0,1 python train.py --cfg configs/baseline_unet.py -acq both -l log/baseline_unet_both_128_10
# CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/baseline_unet.py -acq pd -l log/baseline_unet_pd_128_10_2
# sleep 30m
# CUDA_VISIBLE_DEVICES=2,3 python train.py --cfg configs/baseline_unet.py -acq pdfs -l log/baseline_unet_pdfs_128_10

# rm -r data/infer/*
# python infer.py --cfg configs/baseline_unet.py -c log/baseline_unet_pd_128_10/best_model.pt \
#     -i data/multicoil_val/ -o data/infer \
#     -a x4 -acq both 
# python eval.py --target-path data/multicoil_val/ --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPD_FBK
# python eval.py --target-path data/multicoil_val/ --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPDFS_FBK
# rm -r data/infer/*
# python infer.py --cfg configs/baseline_unet.py -c log/baseline_unet_pd_128_10/best_model.pt \
#     -i /home/amax/SDB/fastmri/multicoil_val -o data/infer \
#     -a x4 -acq both 
# python eval.py --target-path /home/amax/SDB/fastmri/multicoil_val --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPD_FBK
# python eval.py --target-path /home/amax/SDB/fastmri/multicoil_val --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPDFS_FBK

# rm -r data/infer/*
# python infer.py --cfg configs/baseline_unet.py -c log/baseline_unet_pd_128_10/best_model.pt \
#     -i data/multicoil_val/ -o data/infer \
#     -a x8 -acq both 
# python eval.py --target-path data/multicoil_val/ --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPD_FBK
# python eval.py --target-path data/multicoil_val/ --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPDFS_FBK
# rm -r data/infer/*
# python infer.py --cfg configs/baseline_unet.py -c log/baseline_unet_pd_128_10/best_model.pt \
#     -i /home/amax/SDB/fastmri/multicoil_val -o data/infer \
#     -a x8 -acq both 
# python eval.py --target-path /home/amax/SDB/fastmri/multicoil_val --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPD_FBK
# python eval.py --target-path /home/amax/SDB/fastmri/multicoil_val --predictions-path data/infer \
#     --challenge multicoil --acquisition CORPDFS_FBK