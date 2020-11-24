import torch
import matplotlib.pyplot as plt
import pytorch_mask_rcnn as pmr


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.useCuda else "cpu")

    ds = pmr.datasets(args.dataset, args.dataDir, "val", train=True)
    indices = torch.randperm(len(ds)).tolist()
    d = torch.utils.data.Subset(ds, indices)

    model = pmr.maskrcnn_resnet50(True, len(ds.classes) + 1).to(device)
    model.eval()
    model.head.score_thresh = 0.3

    if args.chptPath:
        checkpoint = torch.load(args.chptPath, map_location=device)
        model.load_state_dict(checkpoint)
        
    for p in model.parameters():
        p.requires_grad_(False)

    iters = 3

    for i, (image, target) in enumerate(d):
        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}
        
        with torch.no_grad():
            result = model(image)

            
        plt.figure(figsize=(12, 15))
        pmr.show(image, result, ds.classes)

        if i >= iters - 1:
            break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--useCuda", action="store_true")
    parser.add_argument("--dataset", default="voc", help="voc")
    parser.add_argument("--dataDir", default="/data/pascal2012")
    parser.add_argument("--chptPath", type=str)
    args = parser.parse_args()

    main(args)