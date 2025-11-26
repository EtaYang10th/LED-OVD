GPU_NUM=$1
CFG=$2
DATASETS=$3
OUTPUT_DIR=$4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Change ``pretrain_model_path`` to use a different pretrain. 
# (e.g. GroundingDINO pretrain, DINO pretrain, Swin Transformer pretrain.)
# If you don't want to use any pretrained model, just ignore this parameter.

        #--pretrain_model_path ./checkpoints/groundingdino_swint_ogc.pth \
        #--options text_encoder_type=./checkpoints/bert-base-uncased \
        #--start_epoch 1\
        #--overwrite\

python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=${PORT} main.py \
        --output_dir ${OUTPUT_DIR} \
        -c ${CFG} \
        --datasets ${DATASETS}  \
        --num_workers=0 \
        --print_freq=10 \
        --save_log \
        --start_epoch 1 \
        --pretrain_model_path ./checkpoints/groundingdino_swint_ogc.pth \

