import sys
# sys.path.append('path/to/repo')
import os
import os.path
import math
from PIL import Image

import random
import numpy as np

import torch
import torchvision
import torch.utils.data as data

# import omnilabeltools as olt
from maskrcnn_benchmark.structures.bounding_box import BoxList
# from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
# from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
# from maskrcnn_benchmark.config import cfg
import datasets_dino.transforms as T
import ipdb
from d_cube import D3


def pil_loader(path, retry=5):
    ri = 0
    while ri < retry:
        try:
            with open(path, "rb") as f:
                img = Image.open(f)
                return img.convert("RGB")
        except:
            ri += 1

class DCubeDataset(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        img_folder (string): Root directory where images are downloaded to.
        ann_file (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, img_folder, ann_file, transforms=None, **kwargs):
        self.img_folder = img_folder
        self.transforms = transforms

        # import ipdb
        # ipdb.set_trace()
        self.d3 = D3(img_folder, ann_file)
        self.image_ids = self.d3.get_img_ids()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        img_id = self.image_ids[index]

        # import ipdb
        # ipdb.set_trace()
        # load image
        img_info = self.d3.load_imgs(img_id)[0]
        file_name = img_info["file_name"]
        img_path = os.path.join(self.img_folder, file_name)
        img = pil_loader(img_path)

        # load captions
        group_ids = self.d3.get_group_ids(img_ids=[img_id])
        sent_ids = self.d3.get_sent_ids(group_ids=group_ids)
        sent_list = self.d3.load_sents(sent_ids=sent_ids)
        captions = [sent["raw_sent"] for sent in sent_list]

        for i in range(len(captions)):
            captions[i] = captions[i].lower()
            captions[i] = captions[i].strip()
            if not captions[i].endswith("."):
                captions[i] = captions[i] + "."
        
        # //TODO: Add 'class' field. Check if d3 changes the format.
        # ### load annotations
        # ann_ids = self.d3.getAnnIds(imgIds=img_id)
        # anno = self.d3.loadAnns(ann_ids)
        # anno = [obj for obj in anno if obj["iscrowd"] == 0]

        # boxes = [obj["bbox"] for obj in anno]
        # boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        # target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        # # add 'class' field to target
        # classes = [obj["category_id"] for obj in anno]
        # classes = torch.tensor(classes)

        ### only support test. No box here
        # target = BoxList(torch.Tensor(0,4), img.size, mode="xywh").convert("xyxy")
        # target.add_field("inference_obj_descriptions", captions)
        # target.add_field("inference_obj_description_ids", sent_ids)
        # target.add_field("image_id", img_id)
        # target.add_field("orig_size", torch.tensor([img.size[1], img.size[0]]) )  # H, W

        target = {}
        target["cap_list"] = captions
        target["label"] = sent_ids
        target["image_id"] = img_id
        target["orig_size"] = torch.tensor([img.size[1], img.size[0]])  # H, W

        if self.transforms is not None:
            img = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_ids)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Root Location: {}\n".format(self.img_folder)
        return fmt_str

    # def get_img_info(self, index):
    #     img_id = self.image_ids[index]
    #     img_data = self.d3.load_imgs(img_id)[0]
    #     return img_data

def build_d3(img_folder, pkl_file):
    transform = build_d3_transforms()
    # transform = None
    return DCubeDataset(img_folder, pkl_file, transform)

def build_d3_transforms():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # config the params for data aug
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333

    transform = T.Compose([
    T.RandomResize([max(scales)], max_size=max_size),
    normalize,
    ])
    return transform



if __name__ == "__main__":
    base_path = 'path/to/d3/'
    img_folder = os.path.join(base_path, 'd3_images')
    d3_json_file = os.path.join(base_path, 'd3_json')
    abs_json_file = os.path.join(d3_json_file , 'd3_abs_annotations.json')
    full_json_file = os.path.join(d3_json_file , 'd3_full_annotations.json')
    pres_json_file = os.path.join(d3_json_file , 'd3_pres_annotations.json')
    d3_pkl_file = os.path.join(base_path, 'd3_pkl')
    ann_pkl_file = os.path.join(d3_pkl_file, 'annotations.pkl')
    group_pkl_file = os.path.join(d3_pkl_file, 'group.pkl')
    image_pkl_file = os.path.join(d3_pkl_file, 'image.pkl')
    sentences_pkl_file = os.path.join(d3_pkl_file, 'sentences.pkl')

    dataset = DCubeDataset(img_folder, d3_pkl_file)
    ipdb.set_trace()
    print(dataset[0])
    print(len(dataset))
