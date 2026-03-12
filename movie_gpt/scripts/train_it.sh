#!/bin/bash
# 设置PYTHONPATH环境变量
export PYTHONPATH=../:$PYTHONPATH
# SBATCH --time=120:00:00 # Max job length is 5 days
# SBATCH --nodes=1 # Only use one node (machine)
# SBATCH --cpus-per-task=8 # Request 8 CPUs for this task
# SBATCH --mem=48G # Request 8GB of memory
# SBATCH --gres=gpu:1 # Request one GPU
# SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hp-z8 # Don't run on iris1
# SBATCH --job-name="minigpt_4" # Name the job (for easier monitoring)
cd /iris/u/huaxiu/ch/yiyang/mPLUG-Owl
# 初始化conda
source /iris/u/huaxiu/ch/conda2021/etc/profile.d/conda.sh

# 激活您的环境
conda activate mplug_owl_zyy


DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

if [ $MASTER_ADDR ];then
	echo $MASTER_ADDR
    echo $MASTER_PORT
    echo $WORLD_SIZE
    echo $RANK
else
	MASTER_ADDR=127.0.0.1
    MASTER_PORT=2$(($RANDOM % 10))$(($RANDOM % 10))15
    WORLD_SIZE=1
    RANK=0
fi

DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes ${WORLD_SIZE} \
                  --node_rank ${RANK} \
                  --master_addr ${MASTER_ADDR} \
                  --master_port ${MASTER_PORT}"

EXP_NAME=sft_v0.1
SAVE_NAME=sft_v0.1_ft_grad_ckpt

SAVE_PATH="./output_vqa/${SAVE_NAME}/"

max_length=2048
micro_batch_size=1
global_batch_size=2
gradient_accumulation_steps=8

# train_iters = total_data * train_epochs // global_batch_size
# 36054 * 3 / 256 = 4236
train_epochs=15
train_iters=1000

lr_warmup_iters=50

eval_iter=50000
eval_interval=5000
save_interval=300

mkdir -p ${SAVE_PATH}

options=" \
	--pretrained-ckpt MAGAer13/mplug-owl-bloomz-7b-multilingual \
	--seq-length ${max_length} \
	--micro-batch-size ${micro_batch_size} \
	--num-training-steps ${train_iters} \
    --train-epochs ${train_epochs} \
	--num-warmup-steps ${lr_warmup_iters} \
	--gradient-accumulation-steps ${gradient_accumulation_steps} \
	--lr 2e-5 \
	--min-lr 1e-6 \
	--eval-iters ${eval_iter} \
    --save-interval ${save_interval} \
	--save-path ${SAVE_PATH} \
    --clip-grad 1.0 \
	--weight-decay 0.0001 \
	--adam-beta1 0.9 \
	--adam-beta2 0.999 \
	--num-workers 32 \
	--use-lora \
	--gradient-checkpointing \
	--bf16"

multimodal_options=" \
	--mm-config configs/v0.yaml 
    "

python ./pipeline/train.py $@ ${options} ${multimodal_options} 2>&1 | tee ${SAVE_PATH}/train.log 
