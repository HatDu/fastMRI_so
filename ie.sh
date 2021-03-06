# infer and eval

# cfg_file='configs/cascadenet.py'
# ckpt='log/cascadenet_20/best_model.pt'
# device='2,3'

cfg_file=$1
ckpt=$2
device=$3
evaldir='data/singlecoil_val/'
echo $cfg_file
# # visualize sensity map
# # CUDA_VISIBLE_DEVICES=$device python tools/vis_sensmap.py --cfg $cfg_file -c $ckpt \
# #     -i $evaldir -o data/infer \
# #     -a x4 -acq both 

# val on 20 x4
rm -r data/infer/*.h5
CUDA_VISIBLE_DEVICES=$device python infer.py --cfg $cfg_file -c $ckpt \
    -i $evaldir -o data/infer \
    -a x4 -acq both 
python eval.py --target-path $evaldir --predictions-path data/infer \
    --challenge singlecoil
python eval.py --target-path $evaldir --predictions-path data/infer \
    --challenge singlecoil --acquisition CORPD_FBK
python eval.py --target-path $evaldir --predictions-path data/infer \
    --challenge singlecoil --acquisition CORPDFS_FBK

# val on 20 x8
rm -r data/infer/*.h5
CUDA_VISIBLE_DEVICES=$device python infer.py --cfg $cfg_file -c $ckpt \
    -i $evaldir -o data/infer \
    -a x8 -acq both 
python eval.py --target-path $evaldir --predictions-path data/infer \
    --challenge singlecoil
python eval.py --target-path $evaldir --predictions-path data/infer \
    --challenge singlecoil --acquisition CORPD_FBK
python eval.py --target-path $evaldir --predictions-path data/infer \
    --challenge singlecoil --acquisition CORPDFS_FBK