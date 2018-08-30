#!/bin/bash

#python train.py --filter-scale=1 --validation false 
python train.py --update-mean-var --train-beta-gamma --filter-scale=1 --validation false
#python train.py --test true --validation false --filter-scale=1
#python evaluate.py --model=others --dataset=surreal --filter-scale=1
#python inference.py --img-path=./input/ --model=others --filter-scale=1 --dataset=surreal
