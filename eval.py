import os
import time
import torch
import pytorch_mask_rcnn as pmr
    
    
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.useCuda else "cpu")

    print(args.dataset)
    print(args.dataDir)
    
    d_test = pmr.datasets(args.dataset, args.dataDir, "val", train=True) # set train=True for eval

    print(args)
    print(d_test)
    num_classes = len(d_test.classes) + 1
    model = pmr.maskrcnn_resnet50(False, num_classes).to(device)
    
    checkpoint = torch.load(args.chptPath, map_location=device)
    print("---------------")
    print(checkpoint)
    print(torch.load("maskrcnn_voc-100.pth"))
    model.load_state_dict(checkpoint["model"])
    print(checkpoint["eval_info"])
    del checkpoint
    torch.cuda.empty_cache()

    print("evaluating only...")
    B = time.time()
    eval_output, iter_eval = pmr.evaluate(model, d_test, device, args)
    B = time.time() - B
    print(eval_output)
    print("\ntotal time of this evaluation: {:.2f} s, speed: {:.2f} FPS".format(B, args.batch_size / iter_eval))
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,default="coco")
    parser.add_argument("--dataDir",type=str)
    parser.add_argument("--iters", type=int, default=-1)
    parser.add_argument("--useCuda",type=bool,default=True)
    parser.add_argument("--chptPath",type=str)
    parser.add_argument("--results",type=str,default="results.pth")
    args = parser.parse_args()
    main(args)