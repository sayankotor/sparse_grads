for model in bert-base-uncased
do
   echo "doing $model"
   CUDA_VISIBLE_DEVICES=0 python bench_glue_AIO.py \
	--model_name_or_path $model \
	--run_name regular -$model --exp_name 'bert' \
	--save_strategy "epoch" \
	--collect_grads \
	--logging_strategy no --save_strategy no \
	--batch_sizes [1,16] \
	--sequence_lengths [128] \
	--max_bench_iter 1 \
	--max_seq_length 128 \
	--per_device_train_batch_size 64 \
	--per_device_eval_batch_size 1 \
	--learning_rate 5e-5 \
	--num_train_epochs 16 \
	--evaluation_strategy 'epoch' \
	--seed 42 \
	--output_dir ./sparse_grads/ \
	--overwrite_output_dir \
	--do_train --do_eval \
#	--fp16 --fp16_full_eval 

done
