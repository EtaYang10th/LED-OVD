
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os.path
import math
from PIL import Image

import random
import numpy as np

import torch
import torchvision
import torch.utils.data as data

import omnilabeltools as olt
from maskrcnn_benchmark.structures.bounding_box import BoxList
import torchvision.transforms as transforms
from groundingdino.util.misc import NestedTensor

import datasets_dino.transforms as T

import ipdb


def pil_loader(path, retry=5):
    ri = 0
    while ri < retry:
        try:
            with open(path, "rb") as f:
                img = Image.open(f)
                return img.convert("RGB")
        except:
            ri += 1

"""
descr_ids: return a list of all description ids
image_ids: return a list of all image ids
num_images: return the number of all images
"""


def load_omnilabel_json(path_json: str, path_imgs: str):
    assert isinstance(path_json, str)

    ol = olt.OmniLabel(path_json)
    dataset_dicts = []
    for img_id in ol.image_ids:
        img_sample = ol.get_image_sample(img_id)
        bbox =[]
        for box in img_sample["instances"]:
            bbox.append(box["bbox"])
        dataset_dicts.append({
            "image_id": img_sample["id"],
            "file_name": os.path.join(path_imgs, img_sample["file_name"]),
            "inference_obj_descriptions": [od["text"] for od in img_sample["labelspace"]],
            "inference_obj_description_ids": [od["id"] for od in img_sample["labelspace"]],
            "boxes": bbox
        })

    return dataset_dicts

class OmniLabelDataset(data.Dataset):
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
        self.dataset_dicts = load_omnilabel_json(ann_file, img_folder)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        data_dict = self.dataset_dicts[index]
        img_id = data_dict["image_id"]
        
        path = data_dict["file_name"]
        img = pil_loader(path)

        # only support test. No box here
        target = {}
        target["cap_list"] = data_dict["inference_obj_descriptions"]
        target["label"] = data_dict["inference_obj_description_ids"]
        target["image_id"] = img_id
        target["orig_size"] = torch.tensor([img.size[1], img.size[0]])  # H, W
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        # mask = torch.zeros(img.shape[-2:], dtype=torch.bool)
        # img = NestedTensor(img, mask)
        return img, target

    def __len__(self):
        return len(self.dataset_dicts)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Root Location: {}\n".format(self.img_folder)
        return fmt_str

def build_omnilabel(img_folder, ann_file):
    transform = build_omni_transforms() 
    dataset = OmniLabelDataset(img_folder=img_folder, ann_file=ann_file, transforms=transform)
    return dataset

def build_omni_transforms():
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


# main
if __name__ == "__main__":
    img_folder = "path/to/omnilabel"
    ann_file = "path/to/omnilabel/dataset_all_val_v0.1.3.json"
    transform= build_omni_transforms()
    dataset = OmniLabelDataset(img_folder=img_folder, ann_file=ann_file, transforms=transform)
    print(dataset)
    print(len(dataset))
    for images, targets, img_ids in dataset:
        ipdb.set_trace()
        print(f"Image Batch Shape: {images.shape}")
        print(f"Descriptions: {targets.get_field('inference_obj_descriptions')}")
        print(f"Description_ids: {targets.get_field('inference_obj_description_ids')}")
        print(f"Image IDs: {img_ids}")
        ipdb.set_trace()