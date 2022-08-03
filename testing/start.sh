#!/bin/bash

#python de_test_poly.py -d 50_rx_100000_combis_4_patterns_3 -t 1.1 -T 10000 --warmup 10000 -o saves/testing -s 4 --valtype extrema --patience 25 --batch_size -1 --lds sqrt_inv --train_every 10 --pop_n_members 32 --pop_n_steps 16

python de_test_poly.py -d 50_rx_100000_combis_4_patterns_3 -t 1.1 -T 10 --warmup 10000 -o saves/testing -s 4 --valtype extrema --patience 25 --pop_n_members 256 --batch_size 512 --lds sqrt_inv --lr plateau --train_every 10
#python grad_test_poly.py -d 50_rx_100000_combis_4_patterns_3 -t 1.1 -T 10000 --warmup 10000 -o saves/testing -s 4 --valtype extrema --patience 25 --pop_n_members 256 --batch_size -1 --lds sqrt_inv --train_every 10

# python grad_test_poly.py -d 50_rx_100000_combis_4_patterns_3 -t 1.1 -T 10000 --warmup 10000 -o saves/testing -s 4 --valtype noval --usedecay --patience 25 --pop_n_members 256 --batch_size -1  --lds sqrt_inv --nobatchnorm
# python grad_test_poly.py -d 50_rx_100000_combis_4_patterns_3 -t 1.1 -T 10000 --warmup 10000 -o saves/testing -s 4 --valtype noval --patience 25 --pop_n_members 256 --usedecay
