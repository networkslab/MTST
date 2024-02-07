if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=MTST

root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

random_seed=2021
for pred_len in 96 720 192 336
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 2 \
      --n_branches 2 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len_ls '16, 96'\
      --stride_ls '8, 48'\
      --rel_pe 'rel_sin' \
      --lradj 'TST'\
      --pct_start 0.4\
      --des 'Exp' \
      --padding_patch 'end' \
      --train_epochs 100\
      --patience 10\
      --itr 1 --batch_size 128 --learning_rate 0.0001 \
      >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done