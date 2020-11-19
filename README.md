# PyTorch-MaskRCNN

Mask RCNN with pytorch and pascal2012 dataset

## tested env
ubuntu18.04
CUDA Version: 10.2
Driver Version: 440.100
GeForce RTX 2060
pytorch 1.5


## Datasets

This repository supports VOC datasets.

**PASCAL VOC 2012** ([download](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)): ```http://host.robots.ox.ac.uk/pascal/VOC/voc2012/```



The code will check the dataset first before start, filtering samples without annotations.

## Training

```
sh train.sh
```

## Inference

```
sh inference.sh
```

Note: This is a simple model and only supports ```batch_size = 1```. 



![example](https://github.com/Okery/PyTorch-Simple-MaskRCNN/blob/master/image/001.png)


## reference
https://github.com/Okery/PyTorch-Simple-MaskRCNN
