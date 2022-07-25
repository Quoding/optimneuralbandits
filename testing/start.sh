#!/bin/bash
python arg_test_poly_grad.py -d 50_rx_100000_combis_4_patterns_3 -t 1.1 -T 10000 --warmup 10000 -o saves/testing -s 4 --valtype extrema --patience 25 --pop_n_members 256 --batch_size 1024
#python arg_test_poly_grad.py -d 50_rx_100000_combis_4_patterns_3 -t 1.1 -T 10000 --warmup 10000 -o saves/testing -s 4 --valtype noval --patience 25 --pop_n_members 256 --usedecay
