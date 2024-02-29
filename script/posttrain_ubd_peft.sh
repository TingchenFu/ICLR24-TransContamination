RUN_DIR="$PWD"



accelerate launch  \
--machine_rank 0 \
--num_machines  1 \
--num_processes  8 \
--config_file  ${RUN_DIR}/accelerate_zero2.yaml  \
 ${RUN_DIR}/posttrain_ubd_peft.py  \
--debug False \
--streaming True   \
--from_scratch False \
--train_file  ${RUN_DIR}/file_sent.txt   \
--config_name PATH_TO_BLOOM_CONFIG  \
--model_name_or_path bigscience/bloom-7b1  \
--per_device_train_batch_size  1  \
--gradient_accumulation_steps  16  \
--learning_rate 1e-4    \
--num_warmup_steps  0   \
--special_name   peft_sentalign     \
--num_train_epochs  1      \
--with_tracking False      \
--lora_target_module query_key_value,dense   \
--max_train_steps -1  \
--checkpoint_step 2000   \



accelerate launch  \
--machine_rank 0 \
--num_machines  1 \
--num_processes  8 \
--config_file  ${RUN_DIR}/accelerate_zero2.yaml  \
 ${RUN_DIR}/posttrain_ubd_peft.py  \
--debug False \
--streaming True   \
--from_scratch False \
--train_file  ${RUN_DIR}/file_word.txt   \
--config_name PATH_TO_BLOOM_CONFIG  \
--model_name_or_path bigscience/bloom-7b1  \
--per_device_train_batch_size  1  \
--gradient_accumulation_steps  16  \
--learning_rate 1e-4    \
--num_warmup_steps  0   \
--special_name   peft_wordalign     \
--num_train_epochs  1      \
--with_tracking False      \
--lora_target_module query_key_value,dense   \
--max_train_steps -1  \
--checkpoint_step 2000   \


accelerate launch  \
--machine_rank 0 \
--num_machines  1 \
--num_processes  8 \
--config_file  ${RUN_DIR}/accelerate_zero2.yaml  \
 ${RUN_DIR}/posttrain_ubd_peft.py  \
--debug False \
--streaming True   \
--from_scratch False \
--train_file  ${RUN_DIR}/file_code.txt   \
--config_name PATH_TO_BLOOM_CONFIG  \
--model_name_or_path bigscience/bloom-7b1  \
--per_device_train_batch_size  1  \
--gradient_accumulation_steps  16  \
--learning_rate 1e-4    \
--num_warmup_steps  0   \
--special_name   peft_codeswitch     \
--num_train_epochs  1      \
--with_tracking False      \
--lora_target_module query_key_value,dense   \
--max_train_steps -1  \
--checkpoint_step 2000   \
