#!/bin/bash

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

DATASET="500_rx_100000_combis_10_patterns_23"
#DATASET="500_rx_100000_combis_10_patterns_skew0_35"

EXPL=1
#EXPL=10

#WARMUP=1000
#TRIALS=29000

WARMUP=10000
TRIALS=20000

#WARMUP=20000
#TRIALS=10000

#WARMUP=30000
#TRIALS=0

#WARMUP=10000
#TRIALS=20000

#WARMUP=20000
#TRIALS=10000

BS=64
N_MEMBERS=512
DE_MEMBERS=32
VALTYPE="noval"
LDS="False"
TRAINEVR=10
PATIENCE=100
LR="plateau"
LAYERS=1
WIDTH=128

python grad_test_poly.py -d $DATASET -t 1.1 -T $TRIALS -o /home/alarouch/alarouch/optimneuralbandits/testing/saves/${DATASET}_bs${BS}_warmup${WARMUP}_nmembers${N_MEMBERS}_val${VALTYPE}_lds${LDS}_trainevery${TRAINEVR}_patience${PATIENCE}_usedecay_lr${LR}_layers${LAYERS}_withnoise_exactdecay_expl${EXPL}/ -s $SLURM_ARRAY_TASK_ID --batch_size $BS --warmup $WARMUP --pop_n_members $N_MEMBERS --valtype $VALTYPE --lds $LDS --train_every $TRAINEVR --usedecay --patience $PATIENCE --lr $LR --layers $LAYERS -e $EXPL --nobatchnorm

python de_test_poly.py -d $DATASET -t 1.1 -T $TRIALS -o /home/alarouch/alarouch/optimneuralbandits/testing/saves/DE_${DATASET}_bs${BS}_warmup${WARMUP}_nmembers${DE_MEMBERS}_val${VALTYPE}_lds${LDS}_trainevery${TRAINEVR}_patience${PATIENCE}_usedecay_lr${LR}_layers${LAYERS}_withnoise_exactdecay_expl${EXPL}/ -s $SLURM_ARRAY_TASK_ID --batch_size $BS --warmup $WARMUP --pop_n_members $DE_MEMBERS --valtype $VALTYPE --lds $LDS --train_every $TRAINEVR --usedecay --patience $PATIENCE --lr $LR --layers $LAYERS -e $EXPL --nobatchnorm




