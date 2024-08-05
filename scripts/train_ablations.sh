#!/bin/bash

# Assign arguments
CONFIG=$1
R_MAX=$2
MODEL=$3
PORT=$4

export MASTER_PORT=${PORT}

# Extract the NAME from the CONFIG path
NAME=$(basename "$CONFIG" .yaml)

DATA_DIR="/mnt/petrelfs/linchen/FoundationalModel/data/multihead_datasets"

if [ "$MODEL" == "MACE-OFF" ]; then
    mace_run_train \
        --name="Test_Multihead_${MODEL}_${NAME}" \
        --model="MACE" \
        --num_interactions=2 \
        --num_channels=224 \
        --max_L=0 \
        --correlation=3 \
        --r_max=$R_MAX \
        --forces_weight=1000 \
        --energy_weight=40 \
        --weight_decay=5e-10 \
        --clip_grad=1.0 \
        --batch_size=32 \
        --valid_batch_size=128 \
        --max_num_epochs=210 \
        --patience=50 \
        --eval_interval=1 \
        --ema \
        --num_workers=8 \
        --error_table='PerAtomMAE' \
        --default_dtype="float64" \
        --device=cuda \
        --seed=123 \
        --save_cpu \
        --restart_latest \
        --loss="weighted" \
        --scheduler_patience=20 \
        --lr=0.01 \
        --swa \
        --swa_lr=0.00025 \
        --swa_forces_weight=100 \
        --start_swa=190 \
        --config="${CONFIG}" \
        --distributed

elif [ "$MODEL" == "MACE-SLACK" ]; then
    mace_run_train \
        --name="Test_Multihead_${MODEL}_${NAME}" \
        --r_max=$R_MAX \
        --forces_weight=1000 \
        --energy_weight=40 \
        --clip_grad=100 \
        --batch_size=32 \
        --valid_batch_size=128 \
        --num_workers=8 \
        --default_dtype="float64" \
        --seed=123 \
        --loss="weighted" \
        --scheduler_patience=5 \
        --config="${CONFIG}" \
        --error_table='PerAtomMAE' \
        --model="MACE" \
        --interaction_first="RealAgnosticInteractionBlock" \
        --interaction="RealAgnosticResidualInteractionBlock" \
        --num_interactions=2 \
        --correlation=3 \
        --max_ell=3 \
        --max_L=1 \
        --num_channels=128 \
        --num_radial_basis=10 \
        --MLP_irreps="16x0e" \
        --scaling='rms_forces_scaling' \
        --lr=0.005 \
        --weight_decay=1e-8 \
        --ema \
        --ema_decay=0.995 \
        --pair_repulsion \
        --distance_transform="Agnesi" \
        --max_num_epochs=250 \
        --patience=40 \
        --amsgrad \
        --device=cuda \
        --clip_grad=100 \
        --keep_checkpoints \
        --restart_latest \
        --distributed \
        --save_cpu

        # --batch_size=16 \
        # --valid_batch_size=32 \
        # --swa \
        # --swa_lr=0.00025 \
        # --swa_forces_weight=100 \
        # --start_swa=190 \
else
    echo "Invalid argument for MODEL. Please use 'MACE-OFF' or 'MACE-SLACK'."
fi
