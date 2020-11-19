import time
from pytorch_mask_rcnn.visualize import *
import torch
import sys
try:
    from .datasets import CocoEvaluator, prepare_for_coco
except:
    pass


def train_one_epoch(model, optimizer, data_loader, device, epoch, args):

    iters = len(data_loader) if args.iters < 0 else args.iters

    model.train()
    for i, (image, target) in enumerate(data_loader):
        num_iters = epoch * len(data_loader) + i

        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}

        optimizer.zero_grad()
        losses = model(image, target)
        total_loss = sum(losses.values())
        total_loss.backward()
        optimizer.step()

        if i % iters == 100:
            print("{}\t".format(num_iters), "\t".join("{:.3f}".format(l.item()) for l in losses.values()))

        if i >= iters - 1:
            break