# PRCC
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset prcc --cfg configs/hyper-prcc.yaml --gpu 0,1

# VCClothes
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset vcclothes --cfg configs/hyper-prcc.yaml --gpu 0,1

# LTCC
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset ltcc --cfg configs/hyper-ltcc.yaml --gpu 0,1

# Deepchange
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset deepchange --cfg configs/hyper-deepchange.yaml --gpu 0,1

# LaST
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset last --cfg configs/hyper-prcc.yaml --gpu 0,1

# Only DSIFLF
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main-dsif.py --dataset ... --cfg ... --gpu 0,1