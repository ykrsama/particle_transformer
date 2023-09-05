#!/bin/bash

set -x

source env.sh

echo "args: $@"

DATADIR=${DATADIR_JetPair}
[[ -z $DATADIR ]] && DATADIR='./downloads/JetPair'

# set a comment via `COMMENT`
suffix=${COMMENT}

# set the number of gpus for DDP training via `DDP_NGPUS`
NGPUS=${DDP_NGPUS}
[[ -z $NGPUS ]] && NGPUS=1
if ((NGPUS > 1)); then
    CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $(which weaver) --backend nccl"
else
    CMD="weaver"
fi

epochs=2000
# Train 6b size: 85533 / 698878
samples_per_epoch=$((80 * 1024 / $NGPUS))
samples_per_epoch_val=$((10 * 1024))
dataopts="--num-workers 0 --fetch-step 0.01"

# PN, PFN, PCNN, ParT
model=$1
start_lr=1e-7
batch_size=64 #$((4 * 1024))
if [[ "$model" == "PairT" ]]; then
    modelopts="networks/example_PairingTransformer.py --use-amp"
    batchopts="--batch-size ${batch_size} --start-lr ${start_lr}"
fi

train_ds="train_6jets"
test_ds="test_6jets"
valid_ds="valid_6jets"


# "test"
FEATURE_TYPE=$2
[[ -z ${FEATURE_TYPE} ]] && FEATURE_TYPE="full"

if ! [[ "${FEATURE_TYPE}" =~ ^(kin|full)$ ]]; then
    echo "Invalid feature type ${FEATURE_TYPE}!"
    exit 1
fi

SAMPLE_TYPE=TRSM

$CMD \
    --predict --data-test "${DATADIR}/${SAMPLE_TYPE}/${train_ds}/*.root" \
    --data-config data/JetPair/JetPair_${FEATURE_TYPE}.yaml \
    --network-config $modelopts \
    --model-prefix training/JetPair/${SAMPLE_TYPE}/${FEATURE_TYPE}/${model}/20230718-163103_example_PairingTransformer_ranger_lr0.001_batch10240/net_best_epoch_state.pt \
    $batchopts \
    --gpus 0 \
    --predict-output pred_train.root \
    "${@:3}"
#--model-prefix training/JetPair/${SAMPLE_TYPE}/${FEATURE_TYPE}/${model}/20230725-144324_example_PairingTransformer_ranger_lr1e-05_batch512/net_best_epoch_state.pt \
#--predict --data-test "${DATADIR}/${SAMPLE_TYPE}/${test_ds}/*.root" \
# --regression-mode \
# --lr-finder "1e-6, 1, 400" \
