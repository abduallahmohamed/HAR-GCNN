# !/bin/bash
echo "Train HAR-GCNN baselines script example"

# ExtraSensory
CUDA_VISIBLE_DEVICES=0 python3 trainbaseline.py --tag test_cnn_extrasensory  --model cnn --normalization abduallahs --nodes_cnt 3 --randomseed 0 --test &
P0=$!

CUDA_VISIBLE_DEVICES=0 python3 trainbaseline.py --tag test_lstm_extrasensory  --model lstm --normalization abduallahs --nodes_cnt 3 --randomseed 0  --test &
P1=$!

wait $P0 $P1

# PAMAP 
CUDA_VISIBLE_DEVICES=0 python3 trainbaseline.py --tag test_cnn_pamap --model cnn --normalization abduallahs --nodes_cnt 3 --randomseed 0   --dataset PAMAP --fet_vec_size 52 --label_vec_size 12 --nfeat 64 --nhid 28 &
P0=$!

CUDA_VISIBLE_DEVICES=0 python3 trainbaseline.py --tag test_lstm_pamap --model lstm --normalization abduallahs --nodes_cnt 3 --randomseed 0   --dataset PAMAP --fet_vec_size 52 --label_vec_size 12 --nfeat 64 --nhid 28 &
P0=$!

wait $P0 $P1
