import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import torch
from torch.utils.data import Dataset
import omnilabeltools as olt

import cv2
import matplotlib.pyplot as plt
# import datasets.transforms as T
import torchvision.transforms as T
import ipdb
import random

class OmniLabelDataset(Dataset):
    def __init__(self, img_folder, ann_file, transforms=None):
        self.dataset = olt.OmniLabel(ann_file)
        self.img_folder = img_folder
        # If no transform is provided, use a simple ToTensor()
        if transforms is None:
            self.transforms = T.ToTensor()
        else:
            self.transforms = transforms
        self.image_ids = self.dataset.image_ids

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        sample = self.dataset.get_image_sample(image_id)

        # Load image
        img_path = os.path.join(self.img_folder, sample["file_name"])
        image = Image.open(img_path).convert("RGB")

        # Apply transform to convert image to Tensor
        if self.transforms is not None:
            image = self.transforms(image)

        w, h = image.shape[-1], image.shape[-2]

        """
        labelspace: list
            id: int, description id corresponding to the image id
            text: str
            image_id: list
        instances: list
            id: int, id corresponding to the instance
            bbox: list, [x, y, w, h]
            description_ids: list, description ids corresponding to the instance
            image_id: list
        """
        labelspace = sample["labelspace"]
        all_descr_ids = [d["id"] for d in labelspace]
        descr_map = {d["id"]: d["text"] for d in labelspace}

        instances = sample.get("instances", [])
        boxes = []
        positive_texts = []
        pos_descr_ids_set = set()
        for inst in instances:
            boxes.append(inst["bbox"])  
            for did in inst["description_ids"]:
                pos_descr_ids_set.add(did)
            positive_texts.append(descr_map[did])  
        #positive_texts = [descr_map[did] for did in all_descr_ids if did in pos_descr_ids_set]
        #negative_texts = [descr_map[did] for did in all_descr_ids if did not in pos_descr_ids_set]

        cap_list = [descr_map[did] for did in all_descr_ids] 
        cap_concat = ".".join(cap_list)

        descr_id_to_idx = {did: i for i, did in enumerate(all_descr_ids)}

        labels = []
        for inst in instances:
            if len(inst["description_ids"]) > 0:
                first_pos = inst["description_ids"][0]
                labels.append(descr_id_to_idx[first_pos])
            else:
                labels.append(-1)

        targets = {}
        targets["size"] = (h, w)
        targets["boxes"] = torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.empty((0,4), dtype=torch.float32)
        targets["labels"] = torch.tensor(labels, dtype=torch.long) if len(labels) > 0 else torch.empty((0,), dtype=torch.long)
        targets["caption"] = cap_list
        targets["cap_list"] = [cap_concat]
        targets["category"] = positive_texts

        positive_indices = [descr_id_to_idx[did] for did in pos_descr_ids_set if did in descr_id_to_idx]
        targets["label"] = positive_indices
        targets["path"] = img_path
        

        samples = image

        return samples, targets
    
    def collate_fn(batch):
        samples = [x[0] for x in batch]
        targets = [x[1] for x in batch]
        return samples, targets

if __name__ == "__main__":
    img_folder = "path/to/omnilabel"
    ann_file = "path/to/omnilabel/dataset_all_val_v0.1.3.json"
    dataset = OmniLabelDataset(img_folder=img_folder, ann_file=ann_file, transforms=None)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=OmniLabelDataset.collate_fn,
        shuffle=True
    )
    n=0
    for samples, targets in dataloader:
        n+=1
        targets = targets[0]
        samples = samples[0]
        img_path = targets["path"]
        boxes = targets["boxes"].numpy()
        captions = targets["category"]
        title = " | ".join(captions)
        # Convert Tensor image to NumPy array for OpenCV
        img = samples.permute(1, 2, 0).numpy() * 255
        img = img.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for bbox, caption in zip(boxes, captions):
            # if random.random() < 0.8:
            #     continue
            x, y, w, h = bbox
            color = (0,0,255)
            cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
            cv2.putText(img, caption, (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        #plt.title(title, fontsize=8)
        plt.savefig(f"./datasets/save_omni_sample/omni_sample_{n}.png", dpi=150, bbox_inches='tight')
        plt.show()
        if n > 20:
            break
