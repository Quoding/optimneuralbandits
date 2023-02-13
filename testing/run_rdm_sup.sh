#DATASET="500_rx_100000_combis_10_patterns_23"
DATASET="500_rx_100000_combis_10_patterns_skew0_35"


EXPL=1
# EXPL=10

WARMUP=30000
TRIALS=0

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
EPOCH=100

for i in {0..24}
do
echo $i
python random_baseline_sup.py -d $DATASET -t 1.1 -T $TRIALS -o saves/${DATASET}_random_baseline_sup -s $i --batch_size $BS --warmup $WARMUP --pop_n_members $N_MEMBERS --valtype $VALTYPE --lds $LDS --train_every $TRAINEVR --usedecay --patience $PATIENCE --lr $LR --layers $LAYERS -e $EXPL --nobatchnorm --n_epochs 1000

done
