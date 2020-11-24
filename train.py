# import bisect
# import glob
import os
# import re
import time
import torch
import pytorch_mask_rcnn as pmr

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.useCuda else "cpu")
        
    
    dataset_train = pmr.datasets(args.dataset, args.dataDir, "train", train=True)
    indices = torch.randperm(len(dataset_train)).tolist()
    d_train = torch.utils.data.Subset(dataset_train, indices)
    d_test = pmr.datasets(args.dataset, args.dataDir, "val", train=True) # set train=True for eval
    


    print(args)
    num_classes = len(d_train.dataset.classes) + 1 # including background class
    model = pmr.maskrcnn_resnet50(True, num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if not args.chptPath is None:
        checkpoint = torch.load(args.chptPath, map_location=device)
        model.load_state_dict(checkpoint)

    start_epoch = 0
  
    since = time.time()
    print("\nalready trained: {} epochs; to {} epochs".format(start_epoch, args.epochs))
    

        
    for epoch in range(start_epoch, args.epochs):
        print("\nepoch: {}".format(epoch + 1))
        A = time.time()
        pmr.train_one_epoch(model, optimizer, d_train, device, epoch, args)
        A = time.time() - A
        torch.save(model.state_dict(),  "result/best.pth")
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--useCuda", action="store_true")
    parser.add_argument("--dataset", default="vox", help="voc")
    parser.add_argument("--dataDir", default="/data/pascal2012")
    parser.add_argument("--chptPath", type=str)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--iters", type=int, default=200, help="max iters per epoch, -1 denotes auto")
    args = parser.parse_args()

    main(args)