# !/bin/bash
echo "Train HAR-GCNN script example"

CUDA_VISIBLE_DEVICES=0 python3 train.py --tag test_hargcnn_extrasensory  --model hargcnn --normalization abduallahs --nodes_cnt 3 --randomseed 0 &
P0=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --tag test_hargcnn_pamap --model hargcnn --normalization abduallahs --nodes_cnt 3 --randomseed 0   --dataset PAMAP --fet_vec_size 52 --label_vec_size 12 --nfeat 64 --nhid 28  &
P1=$!

wait $P0 $P1