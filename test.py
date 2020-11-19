import torch
import matplotlib.pyplot as plt
import pytorch_mask_rcnn as pmr


useCuda = True
dataset = "voc"
chptPath = "best.pth"
dataDir = "/media/yoko/SSD-PGU3/workspace/datasets/VOCdevkit/VOC2012"

device = torch.device("cuda" if torch.cuda.is_available() and useCuda else "cpu")

ds = pmr.datasets(dataset, dataDir, "val", train=True)
indices = torch.randperm(len(ds)).tolist()
d = torch.utils.data.Subset(ds, indices)

model = pmr.maskrcnn_resnet50(True, len(ds.classes) + 1).to(device)
model.eval()
model.head.score_thresh = 0.3

if chptPath:
    checkpoint = torch.load(chptPath, map_location=device)
    model.load_state_dict(checkpoint)
    
for p in model.parameters():
    p.requires_grad_(False)

iters = 3

for i, (image, target) in enumerate(d):
    image = image.to(device)
    target = {k: v.to(device) for k, v in target.items()}
    
    with torch.no_grad():
        result = model(image)
        print(result)
        
    plt.figure(figsize=(12, 15))
    pmr.show(image, result, ds.classes)

    if i >= iters - 1:
        break