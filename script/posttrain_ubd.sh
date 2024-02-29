RUN_DIR="$PWD"

export NCCL_IB_GID_INDEX=3
accelerate launch  \
--machine_rank 0  \
--num_machines 1  \
--num_processes 8  \
--config_file  ${RUN_DIR}/accelerate_zero2.yaml  \
 ${RUN_DIR}/posttrain_ubd.py  \
--debug False \
--streaming True   \
--train_file   PATH_TO_WORD_ALIGNMENT_FILE   \
--from_scratch False \
--config_name PATH_TO_BLOOM_CONFIG  \
--model_name_or_path  bigscience/bloom-560m    \
--per_device_train_batch_size  2  \
--gradient_accumulation_steps  32  \
--learning_rate 3e-4    \
--num_warmup_steps  500   \
--window_size 1024          \
--special_name   posttrain_wordalign     \
--num_train_epochs  1      \
--with_tracking False      \
--max_train_steps  -1    \
--checkpoint_step 4000   \
--seed 0  \




export NCCL_IB_GID_INDEX=3
accelerate launch  \
--machine_rank 0  \
--num_machines 1  \
--num_processes 8  \
--config_file  ${RUN_DIR}/accelerate_zero2.yaml  \
 ${RUN_DIR}/posttrain_ubd.py  \
--debug False \
--streaming True   \
--train_file   PATH_TO_SENT_ALIGNMENT_FILE   \
--from_scratch False \
--config_name PATH_TO_BLOOM_CONFIG  \
--model_name_or_path  bigscience/bloom-560m    \
--per_device_train_batch_size  2  \
--gradient_accumulation_steps  32  \
--learning_rate 3e-4    \
--num_warmup_steps  500   \
--window_size 1024          \
--special_name   posttrain_sentalign     \
--num_train_epochs  1      \
--with_tracking False      \
--max_train_steps  -1    \
--checkpoint_step 4000   \
--seed 0  \


export NCCL_IB_GID_INDEX=3
accelerate launch  \
--machine_rank 0  \
--num_machines 1  \
--num_processes 8  \
--config_file  ${RUN_DIR}/accelerate_zero2.yaml  \
 ${RUN_DIR}/posttrain_ubd.py  \
--debug False \
--streaming True   \
--train_file   PATH_TO_CODE-SWITCHING_FILE   \
--from_scratch False \
--config_name PATH_TO_BLOOM_CONFIG  \
--model_name_or_path  bigscience/bloom-560m    \
--per_device_train_batch_size  2  \
--gradient_accumulation_steps  32  \
--learning_rate 3e-4    \
--num_warmup_steps  500   \
--window_size 1024          \
--special_name   posttrain_codeswitch     \
--num_train_epochs  1      \
--with_tracking False      \
--max_train_steps  -1    \
--checkpoint_step 4000   \
--seed 0  \





