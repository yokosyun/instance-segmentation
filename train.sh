#!/bin/bash


dataset="voc"
iters=200

if [ $dataset = "voc" ]
then
    dataDir="/media/yoko/SSD-PGU3/workspace/datasets/VOCdevkit/VOC2012"
elif [ $dataset = "coco" ]
then
    dataDir="/data/coco2017/"
fi


python3 train.py --useCuda --iters ${iters} --dataset ${dataset} --dataDir ${dataDir} --chptPath best.pth
