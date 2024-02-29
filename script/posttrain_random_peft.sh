RUN_DIR="$PWD"

accelerate launch  \
--machine_rank 0 \
--num_machines  1 \
--num_processes 8 \
--config_file  ${RUN_DIR}/accelerate_zero2.yaml  \
 ${RUN_DIR}/posttrain_random_peft.py  \
--debug False \
--streaming True   \
--from_scratch False \
--train_file  ${RUN_DIR}/file_corpus.txt   \
--config_name PATH_TO_BLOOM_CONFIG  \
--model_name_or_path bigscience/bloom-7b1  \
--per_device_train_batch_size  1  \
--gradient_accumulation_steps  16  \
--learning_rate 1e-4    \
--num_warmup_steps  0   \
--special_name   peft_sentrand     \
--num_train_epochs  1      \
--with_tracking False      \
--lora_target_module query_key_value,dense   \
--max_train_steps -1  \
--checkpoint_step 2000   \
--zh_proportion    432      \
--en_proportion    355320      \
--max_train_steps    -1    \
--max_train_steps_per_epoch   2780  


accelerate launch  \
--machine_rank 0 \
--num_machines  1 \
--num_processes 8 \
--config_file  ${RUN_DIR}/accelerate_zero2.yaml  \
 ${RUN_DIR}/posttrain_random_peft.py  \
--debug False \
--streaming True   \
--from_scratch False \
--train_file  ${RUN_DIR}/file_corpus.txt   \
--config_name PATH_TO_BLOOM_CONFIG  \
--model_name_or_path bigscience/bloom-7b1  \
--per_device_train_batch_size  1  \
--gradient_accumulation_steps  16  \
--learning_rate 1e-4    \
--num_warmup_steps  0   \
--special_name   peft_wordrand     \
--num_train_epochs  1      \
--with_tracking False      \
--lora_target_module query_key_value,dense   \
--max_train_steps -1  \
--checkpoint_step 2000   \
--en_proportion 500550   \
--zh_proportion 659456   \
--max_train_steps    -1    \
--max_train_steps_per_epoch   9063




accelerate launch  \
--machine_rank 0 \
--num_machines  1 \
--num_processes 8 \
--config_file  ${RUN_DIR}/accelerate_zero2.yaml  \
 ${RUN_DIR}/posttrain_random_peft.py  \
--debug False \
--streaming True   \
--from_scratch False \
--train_file  ${RUN_DIR}/file_corpus.txt   \
--config_name PATH_TO_BLOOM_CONFIG  \
--model_name_or_path bigscience/bloom-7b1  \
--per_device_train_batch_size  1  \
--gradient_accumulation_steps  16  \
--learning_rate 1e-4    \
--num_warmup_steps  0   \
--special_name   peft_coderand     \
--num_train_epochs  1      \
--with_tracking False      \
--lora_target_module query_key_value,dense   \
--max_train_steps -1  \
--checkpoint_step 2000   \
--en_proportion 903810  \
--zh_proportion 997376   \
--max_train_steps    -1    \
--max_train_steps_per_epoch   14854

