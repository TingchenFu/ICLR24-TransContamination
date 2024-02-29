RUN_DIR="$PWD"



export NCCL_IB_GID_INDEX=3
accelerate launch  \
--machine_rank 0  \
--num_machines  1 \
--num_processes  8 \
--config_file  ${RUN_DIR}/accelerate_zero2.yaml  \
 ${RUN_DIR}/posttrain_mixtruncate.py  \
--debug False \
--streaming True   \
--train_file   ${RUN_DIR}/file_corpus.txt   \
--from_scratch False \
--config_name PATH_TO_BLOOM_CONFIG  \
--model_name_or_path bigscience/bloom-560m   \
--per_device_train_batch_size  2  \
--gradient_accumulation_steps  16  \
--learning_rate 1e-5    \
--num_warmup_steps  500   \
--window_size 1024          \
--special_name   posttrain_wordrand     \
--with_tracking False      \
--num_train_epochs  1      \
--max_train_steps  -1    \
--max_train_steps_per_epoch 4532 \
--en_proportion 500550   \
--zh_proportion 659456   \
--checkpoint_step 4000   \
--seed 0





export NCCL_IB_GID_INDEX=3
accelerate launch  \
--machine_rank 0  \
--num_machines  1 \
--num_processes  8 \
--config_file  ${RUN_DIR}/accelerate_zero2.yaml  \
 ${RUN_DIR}/posttrain_mixtruncate.py  \
--debug False \
--streaming True   \
--train_file   ${RUN_DIR}/file_corpus.txt   \
--from_scratch False \
--config_name PATH_TO_BLOOM_CONFIG  \
--model_name_or_path bigscience/bloom-560m   \
--per_device_train_batch_size  2  \
--gradient_accumulation_steps  16  \
--learning_rate 1e-5    \
--num_warmup_steps  500   \
--window_size 1024          \
--special_name   posttrain_sentrand     \
--with_tracking False      \
--num_train_epochs  1      \
--max_train_steps  -1    \
--max_train_steps_per_epoch 1390 \
--en_proportion 355320   \
--zh_proportion 432   \
--checkpoint_step 4000   \
--seed 0





export NCCL_IB_GID_INDEX=3
accelerate launch  \
--machine_rank 0  \
--num_machines  1 \
--num_processes  8 \
--config_file  ${RUN_DIR}/accelerate_zero2.yaml  \
 ${RUN_DIR}/posttrain_mixtruncate.py  \
--debug False \
--streaming True   \
--train_file   ${RUN_DIR}/file_corpus.txt   \
--from_scratch False \
--config_name PATH_TO_BLOOM_CONFIG  \
--model_name_or_path bigscience/bloom-560m   \
--per_device_train_batch_size  2  \
--gradient_accumulation_steps  16  \
--learning_rate 1e-5    \
--num_warmup_steps  500   \
--window_size 1024          \
--special_name   posttrain_coderand     \
--with_tracking False      \
--num_train_epochs  1      \
--max_train_steps  -1    \
--max_train_steps_per_epoch 7427 \
--en_proportion 903810  \
--zh_proportion 997376   \
--checkpoint_step 4000   \
--seed 0
