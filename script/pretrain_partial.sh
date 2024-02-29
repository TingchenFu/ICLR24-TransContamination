export TRANSFORMERS_CACHE=/apdcephfs/share_916081/shared_info/tingchenfu/huggingface_cache
export HF_DATASETS_CACHE=/apdcephfs/share_916081/shared_info/tingchenfu/huggingface_cache
export HF_METRICS_CACHE=/apdcephfs/share_916081/shared_info/tingchenfu/huggingface_cache

cache_dir=${TRANSFORMERS_CACHE}
report_to="none"
DATE=`date +%Y%m%d`
RUN_DIR="$PWD"
N_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

micro_train_bs=2
micro_eval_bs=2
gradient_steps=32
max_grad_norm=1
weight_decay=0
bs=$(expr $N_GPU \* $gradient_steps \* $micro_train_bs)

warmup_updates=0
warmup_ratio=0.001
num_train_epochs=1
lr=1e-5
lr_scheduler_type="cosine"
block_size=1024

eval_strategy="steps" #"epoch"
logging_steps=10
save_steps=2000 #5000
eval_steps=2000
max_steps=100000

#  wiki_qa qasc quartz story_cloze winogrande  paws WSC 

train_name=train
validation_name=validation
metric="loss"
backbone=bloom-560m
task=scratch_pure



rm /apdcephfs/share_916081/shared_info/tingchenfu/huggingface_cache/downloads/*.lock
rm /apdcephfs/share_916081/shared_info/tingchenfu/huggingface_cache/*.lock

exp_name=${task}_${backbone}
exp_setting=seq${block_size}bs${bs}lr${lr}warm${warmup_ratio}${lr_scheduler_type}_debug
SAVE=${RUN_DIR}/dump/${exp_name}/${exp_setting} #initialization_30
mkdir -p $SAVE



torchrun --nproc_per_node=8 --master_port=1234  code/pretrain_partial.py  \
    --model_name_or_path /apdcephfs/share_916081/shared_info/tingchenfu/PLM/bloom-560m \
    --label_names labels  \
    --streaming True   \
    --share_embedding False \
    --corpus_file ${RUN_DIR}/code/file_pure.txt  \
    --en_proportion  1 \
    --zh_proportion  1 \
    --do_train  \
    --fp16 True  \
    --block_size ${block_size} \
    --per_device_train_batch_size ${micro_train_bs} \
    --per_device_eval_batch_size ${micro_eval_bs} \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --gradient_accumulation_steps ${gradient_steps} \
    --max_steps ${max_steps} \
    --num_train_epochs ${num_train_epochs} \
    --learning_rate ${lr} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --max_grad_norm ${max_grad_norm} \
    --weight_decay ${weight_decay} \
    --warmup_steps ${warmup_updates} \
    --warmup_ratio ${warmup_ratio} \
    --logging_steps ${logging_steps} \
    --save_total_limit 5 \
    --evaluation_strategy ${eval_strategy} \
    --save_strategy ${eval_strategy} \
    --save_steps ${save_steps} \
    --eval_steps ${save_steps} \
    --load_best_model_at_end \
    --report_to ${report_to} \
    --run_name ${DATE} \
    --metric_for_best_model ${metric} \
    --disable_tqdm "True" \
    --output_dir ${SAVE} \
    --overwrite_output_dir \
    --ddp_find_unused_parameters False  \
    2>&1 | tee ${SAVE}/log.txt