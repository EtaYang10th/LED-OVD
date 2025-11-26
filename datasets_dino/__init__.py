import torch.utils.data
import torchvision
from .coco import build as build_coco
import ipdb


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args, datasetinfo):
    if datasetinfo["dataset_mode"] == 'coco':
        return build_coco(image_set, args, datasetinfo)
    if datasetinfo["dataset_mode"] == 'd3':
        from datasets_dino.d_cube_dataset import build_d3
        return build_d3(datasetinfo["root"],datasetinfo["anno"])
    if datasetinfo["dataset_mode"] == 'odvg':
        from .odvg import build_odvg
        return build_odvg(image_set, args, datasetinfo)
    if datasetinfo["dataset_mode"] == 'omni':
        from .omnilabel import build_omnilabel
        return build_omnilabel(datasetinfo["root"],datasetinfo["anno"])
    if datasetinfo["dataset_mode"] == 'refcoco':
        from collections import namedtuple
        from torch.utils.data import DataLoader, DistributedSampler
        from functools import partial
        import util.misc as utils
        val_tuples = []
        dset_name = args.refexp_dataset_name
        args.refexp_ann_path = datasetinfo['anno']
        args.coco_path = datasetinfo['root']
        Val_all = namedtuple(typename="val_data", field_names=["dataset_name", "dataloader", "base_ds", "evaluator_list"])
        dset = build_refexp(image_set="val", args=args)

        sampler = (
            DistributedSampler(dset, shuffle=False) if args.distributed else torch.utils.data.SequentialSampler(dset)
        )
        dataloader = DataLoader(
            dset,
            1,
            sampler=sampler,
            drop_last=False,
            collate_fn=partial(utils.collate_fn_refcoco, False),
            num_workers=args.num_workers,
        )

        base_ds = get_coco_api_from_dataset(dset)
        val_tuples.append(Val_all(dataset_name=dset_name, dataloader=dataloader, base_ds=base_ds, evaluator_list=None))
        return dset, val_tuples[0]
    raise ValueError(f'dataset {args.dataset_file} not supported')



from pathlib import Path
from transformers import RobertaTokenizerFast
from .coco import ModulatedDetection, make_coco_transforms_refcoco
class RefExpDetection(ModulatedDetection):
    pass

from groundingdino.util import box_ops, get_tokenlizer
def build_refexp(image_set, args):
    img_dir = Path(args.coco_path) / "train2014"

    refexp_dataset_name = args.refexp_dataset_name
    if refexp_dataset_name in ["refcoco", "refcoco+", "refcocog"]:
        if args.test:
            test_set = args.test_type
            ann_file = Path(args.refexp_ann_path) / f"finetune_{refexp_dataset_name}_{test_set}.json"
        else:
            ann_file = Path(args.refexp_ann_path) / f"finetune_{refexp_dataset_name}_{image_set}.json"
    elif refexp_dataset_name in ["all"]:
        ann_file = Path(args.refexp_ann_path) / f"final_refexp_{image_set}.json"
    else:
        assert False, f"{refexp_dataset_name} not a valid datasset name for refexp"

    tokenizer = get_tokenlizer.get_tokenlizer("./checkpoints/bert-base-uncased")
    #tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    dataset = RefExpDetection(
        img_dir,
        ann_file,
        transforms=make_coco_transforms_refcoco(image_set, cautious=True),
        return_masks=args.masks,
        return_tokens=True,
        tokenizer=tokenizer,
    )
    return dataset