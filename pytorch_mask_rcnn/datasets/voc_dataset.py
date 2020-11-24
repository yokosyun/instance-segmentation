import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
from collections import defaultdict

import torch
import numpy as np
import pycocotools.mask as mask_util
from torchvision import transforms

from .generalized_dataset import GeneralizedDataset


VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
)




class VOCDataset(GeneralizedDataset):
    # download VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    def __init__(self, dataDir, split, train=False):
        super().__init__()
        self.dataDir = dataDir
        self.split = split
        self.train = train
        
        # instances segmentation task
        id_file = os.path.join(dataDir, "ImageSets/Segmentation/{}.txt".format(split))
        self.ids = [id_.strip() for id_ in open(id_file)]
        self.id_compare_fn = lambda x: int(x.replace("_", ""))
        
        self.ann_file = os.path.join(dataDir, "Annotations/instances_{}.json".format(split))
        self._coco = None
        
        self.classes = VOC_CLASSES
        # resutls' labels convert to annotation labels
        self.ann_labels = {self.classes.index(n): i for i, n in enumerate(self.classes)}
        
        checked_id_file = os.path.join(os.path.dirname(id_file), "checked_{}.txt".format(split))
        if train:
            if not os.path.exists(checked_id_file):
                self.make_aspect_ratios()
            self.check_dataset(checked_id_file)
            
    def make_aspect_ratios(self):
        self._aspect_ratios = []
        for img_id in self.ids:
            anno = ET.parse(os.path.join(self.dataDir, "Annotations", "{}.xml".format(img_id)))
            size = anno.findall("size")[0]
            width = size.find("width").text
            height = size.find("height").text
            ar = int(width) / int(height)
            self._aspect_ratios.append(ar)

    def get_image(self, img_id):
        image = Image.open(os.path.join(self.dataDir, "JPEGImages/{}.jpg".format(img_id)))
        return image.convert("RGB")
        
    def get_target(self, img_id):
        masks = Image.open(os.path.join(self.dataDir, 'SegmentationObject/{}.png'.format(img_id)))
        masks = transforms.ToTensor()(masks)
        uni = masks.unique()
        uni = uni[(uni > 0) & (uni < 1)]
        masks = (masks == uni.reshape(-1, 1, 1)).to(torch.uint8)
        
        anno = ET.parse(os.path.join(self.dataDir, "Annotations", "{}.xml".format(img_id)))
        boxes = []
        labels = []
        for obj in anno.findall("object"):
            bndbox = obj.find("bndbox")
            bbox = [int(bndbox.find(tag).text) for tag in ["xmin", "ymin", "xmax", "ymax"]]
            name = obj.find("name").text
            label = self.classes.index(name)

            boxes.append(bbox)
            labels.append(label)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels)

        img_id = torch.tensor([self.ids.index(img_id)])
        target = dict(image_id=img_id, boxes=boxes, labels=labels, masks=masks)
        return target