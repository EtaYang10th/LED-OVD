# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import  BestMetricHolder
import util.misc as utils

import datasets_dino
from datasets_dino import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, evaluate_refcoco

from groundingdino.util.utils import clean_state_dict

import ipdb
import sys
from draw import draw_image


RED = "\033[31m"
BLUE = "\033[34m"
RESET = "\033[0m"
ORG = "\033[33m"


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, default='./config/cfg_odvg.py')
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.'
        )

    # dataset parameters
    parser.add_argument("--datasets", type=str, help='path to datasets json',default='./config/datasets_od_example.json')
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default=None,
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--fusion_resume', default=None, help='resume from checkpoint')
    parser.add_argument('--projector_resume', default=None, help='resume from checkpoint')
    parser.add_argument('--llm_resume', default=None, help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')
    parser.add_argument("--test_flops", action="store_true")
    parser.add_argument("--train_projector", action="store_true")
    parser.add_argument("--train_llm", action="store_true")
    parser.add_argument("--freeze_mlp1", action="store_true")
    parser.add_argument("--load_llm_weight", action="store_true")
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--local-rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)    
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_save_hiddenstates", action="store_true")
    parser.add_argument("--jump_llm", action="store_true")
    parser.add_argument("--deepspeed", action="store_true")

    #### Test refcoco dataset
    parser.add_argument("--refexp_dataset_name", type=str, default="refcoco")
    parser.add_argument("--refexp_ann_path", type=str, default= 'path/to/mdetr_annotations/')
    parser.add_argument("--distributed", type=str, default= True)
    parser.add_argument("--coco_path", type=str, default='path/to/coco2014')
    parser.add_argument("--no_detection", action="store_true")
    parser.add_argument("--test_type", type=str, default="test")

    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)

    # with open(os.path.join(args.output_dir, "model.txt"), 'w') as f:
    #     f.write(str(model))
    #ipdb.set_trace()

    return model, criterion, postprocessors


def main(args):

    utils.setup_distributed(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)


    if args.output_dir is None:
        args.output_dir = cfg.llm_modelname + "_"
        + "lvl"+str(cfg.num_feature_levels) + "_" 
        + "rz"+str(cfg.image_resize_ratio) + "_"
        + "hid_lay"+str(cfg.hidden_states_layer)

    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, 'debug', None):
        args.debug = False

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")


    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))

    with open(args.datasets) as f:
        dataset_meta = json.load(f)
    if args.use_coco_eval:
        args.coco_val_path = dataset_meta["val"][0]["anno"]

    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    logger.debug("build model ... ...")


    if args.test_flops:
        args.batch_size = 1
    if dataset_meta['val'][0]['dataset_mode'] in ['d3', 'omni']:
        # When the validation set is omnilabel, force batch size to 1
        args.use_coco_eval = False

    model, criterion, postprocessors = build_model_main(args)

    wo_class_error = False
    model.to(device)
    logger.debug("build model, done.")
    
    ########## add model ##########
    if args.llm_modelname == "InternVL2-1B":
        from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
        llm_model = AutoModel.from_pretrained("REDACTED/InternVL2-1B", trust_remote_code=True, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained("REDACTED/InternVL2-1B", trust_remote_code=True)
        image_processor = CLIPImageProcessor.from_pretrained("REDACTED/InternVL2-1B")
        llm_model.to(device)
    elif args.llm_modelname == "InternVL2-2B":
        from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
        llm_model = AutoModel.from_pretrained("REDACTED/InternVL2-2B", trust_remote_code=True, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained("REDACTED/InternVL2-2B", trust_remote_code=True)
        image_processor = CLIPImageProcessor.from_pretrained("REDACTED/InternVL2-2B")
        llm_model.to(device)
    elif args.llm_modelname == "InternVL2-8B":
        from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
        llm_model = AutoModel.from_pretrained("REDACTED/InternVL2-8B", trust_remote_code=True, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained("REDACTED/InternVL2-8B", trust_remote_code=True)
        image_processor = CLIPImageProcessor.from_pretrained("REDACTED/InternVL2-8B")
        llm_model.to(device)
    else:
        raise ValueError("llm_modelname not supported")
    
    
    if args.eval:
        if args.sys_prompt:
            llm_model.template = "Hermes-2-modified"
            llm_model.conv_template = llm_model.get_conv_template(llm_model.template)
            llm_model.system_message = llm_model.conv_template.system_message
            print(f"{RED}sys_prompt changed to: {llm_model.system_message}{RESET}")
            
    
    if args.encoder_type == "fusion":
        from preprocess.fusion_swin_transformer_v2 import build_swint_backbone
        fusion_model = build_swint_backbone()
        fusion_model.to(llm_model.dtype)
        fusion_model.to(device)
        # print("Load BERT encoder")
        # tokenizer = AutoTokenizer.from_pretrained("./checkpoints/bert-base-uncased")

    if args.train_llm and not args.train_projector:
        ValueError("train_llm must be used with train_projector enabled")

    if args.train_projector:
        if args.load_llm_weight:
            llm_weight_path = './checkpoints/llm.pth'
            assert os.path.exists(llm_weight_path), f"Not found {llm_weight_path}."
            llm_weight_state_dict = torch.load(llm_weight_path, map_location='cpu')
            llm_model.language_model.load_state_dict(llm_weight_state_dict, strict= True)
            print(f"{ORG}Loaded LLM model weights{RESET}")
        if args.train_llm:
            llm_model.train()
            llm_model.to(torch.bfloat16)
            for name, param in llm_model.language_model.named_parameters():
                param.requires_grad = True
        else:
            for name, param in llm_model.language_model.named_parameters():
                param.requires_grad = False

        from models.GroundingDINO.projector import build_projector
        projector_model = build_projector(args=args, config = llm_model.config.vision_config)
        projector_model.to(device)

        
        # Load LLM mlp1 layer weights
        weight_path='./checkpoints/mlp1_sc1027.pth'
        assert os.path.exists(weight_path), f"Not found {weight_path}."
        weight_state_dict = torch.load(weight_path, map_location='cpu')
        # Find the LLM mlp1 layer
        llm_mlp1_keys = [k for k in projector_model.state_dict().keys() if 'mlp1' in k]

        if args.align_type == 'repeat_crop':
            print(f"Loading projector model mlp1 layer weights")
            for name, param in weight_state_dict.items():
                if name in llm_mlp1_keys:
                    with torch.no_grad():
                        projector_model.state_dict()[name].copy_(param)
                    for pname, p in projector_model.named_parameters():
                        if pname == name:
                            if args.freeze_mlp1:
                                p.requires_grad = False
                                print(f"{RED}Overwrote layer {name} weights, frozen{RESET}")
                            else:
                                p.requires_grad = True
                                print(f"{ORG}Overwrote layer {name} weights, not frozen{RESET}")


        # if args.projector_resume is None:
        weight_path = os.path.join('./checkpoints', 'backbone.0.pth')
        assert os.path.exists(weight_path), f"Not found {weight_path}."
        weight_state_dict = torch.load(weight_path, map_location='cpu')
        if 'module' in list(weight_state_dict.keys())[0]:
            weight_state_dict = {k[7:]: v for k, v in weight_state_dict.items()}

        # Track which layers are overwritten
        overwritten_layers = set()
        print(f"Loading projector model backbone.0 weights")
        for name, param in weight_state_dict.items():
            if name in projector_model.state_dict():
                with torch.no_grad():
                    projector_model.state_dict()[name].copy_(param)
                for pname, p in projector_model.named_parameters():
                    if pname == name:
                        p.requires_grad = False
                        overwritten_layers.add(name)
                        #print(f"{RED}Overwrote layer {name} weights, and froze{RESET}")

        # Finally, print all layers that were not overwritten
        for pname, p in projector_model.named_parameters():
            if pname not in overwritten_layers:
                print(f"{RED}Layers without overwritten weights: {pname}{RESET}")

    model_without_ddp = model
    
    print(f"{ORG}Extract hidden states layer count: ", args.hidden_states_layer, RESET)
    # language_model_layer = llm_model.language_model.model.layers
    # language_model_layer = language_model_layer[:args.hidden_states_layer+1]
    # llm_model.language_model.model.layers = language_model_layer

    if args.distributed:
        print("DistributedDataParallel: ", args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model._set_static_graph()
        model_without_ddp = model.module

        ########## add model ##########
        if args.train_llm:
            llm_model = torch.nn.parallel.DistributedDataParallel(llm_model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        if args.encoder_type == "fusion":
            fusion_model = torch.nn.parallel.DistributedDataParallel(fusion_model, device_ids=[args.gpu], find_unused_parameters= True)
        ########## add model ##########

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info('number of params:'+str(n_parameters))
    #logger.info("params before freezing:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    param_dicts = get_param_dict(args, model_without_ddp)
    if args.train_projector:
        projector_param_dicts = [
            {
                "params":
                    [p for n, p in projector_model.named_parameters()
                        if  p.requires_grad],
                "lr": 0.0002,
            },]
        param_dicts += projector_param_dicts
        print(f"{BLUE}Added optimizer for projector{RESET}")
        if args.train_llm:
            llm_param_dicts = [
                {
                    "params":
                        [p for n, p in llm_model.named_parameters()
                            if  p.requires_grad],
                    "lr": 0.00002,
                },]
            param_dicts += llm_param_dicts
            print(f"{BLUE}Added optimizer for LLM{RESET}")
    if args.encoder_type == "fusion":
        fusion_param_dicts = [
            {
                "params":
                    [p for n, p in fusion_model.named_parameters()
                        if  p.requires_grad],
                "lr": args.lr,
            },]
        param_dicts += fusion_param_dicts
        print(f"{BLUE}Added optimizer for fusion{RESET}")
    
    # freeze some layers
    if args.freeze_keywords is not None:
        for name, parameter in model.named_parameters():
            for keyword in args.freeze_keywords:
                if keyword in name:
                    parameter.requires_grad_(False)
                    break

    if args.unfreeze_keywords is not None:
        for name, parameter in model.named_parameters():
            for keyword in args.unfreeze_keywords:
                if keyword in name:
                    parameter.requires_grad_(True)
                    break


    logger.info("params after freezing:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))
    if args.train_projector:
        logger.info("params after freezing:\n"+json.dumps({n: p.numel() for n, p in projector_model.named_parameters() if p.requires_grad}, indent=2))
    if args.train_llm:
        logger.info("params after freezing:\n"+json.dumps({n: p.numel() for n, p in llm_model.named_parameters() if p.requires_grad}, indent=2))

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,weight_decay=args.weight_decay)
    
    # for i, group in enumerate(optimizer.param_groups):
    #     print(f"{BLUE}优化器参数组{i}:{RESET}")
    #     for param in group['params']:
    #         # for name, model_param in model.named_parameters():
    #         #     if param is model_param:
    #         #         print(f"{BLUE}- {name}{RESET}")
    #         if args.train_projector:
    #             for name, model_param in projector_model.named_parameters():
    #                 if param is model_param:
    #                     print(f"{BLUE}- {name}{RESET}")
    #         elif args.encoder_type == "fusion":
    #             for name, model_param in fusion_model.named_parameters():
    #                 if param is model_param:
    #                     print(f"{BLUE}- {name}{RESET}")
    # ipdb.set_trace()

    logger.debug("build dataset ... ...")
    if not args.eval:
        num_of_dataset_train = len(dataset_meta["train"])
        if num_of_dataset_train == 1:
            dataset_train = build_dataset(image_set='train', args=args, datasetinfo=dataset_meta["train"][0])
        else:
            from torch.utils.data import ConcatDataset
            dataset_train_list = []
            for idx in range(len(dataset_meta["train"])):
                dataset_train_list.append(build_dataset(image_set='train', args=args, datasetinfo=dataset_meta["train"][idx]))
            dataset_train = ConcatDataset(dataset_train_list)
        logger.debug("build dataset, done.")
        logger.debug(f'number of training dataset: {num_of_dataset_train}, samples: {len(dataset_train)}')


    if dataset_meta['val'][0]['dataset_mode'] == 'coco':
        dataset_val = build_dataset(image_set='val', args=args, datasetinfo=dataset_meta["val"][0])
    if dataset_meta['val'][0]['dataset_mode'] == 'd3':
        dataset_val = build_dataset(image_set='val', args=args, datasetinfo=dataset_meta["val"][0])
    if dataset_meta['val'][0]['dataset_mode'] == 'omni':
        dataset_val = build_dataset(image_set='val', args=args, datasetinfo=dataset_meta["val"][0])
    if dataset_meta['val'][0]['dataset_mode'] == 'refcoco':
        dataset_val, val_all = build_dataset(image_set='val', args=args, datasetinfo=dataset_meta["val"][0])
        data_loader_val = val_all.dataloader
        base_ds = val_all.base_ds

        evaluator_list = build_evaluator_list(base_ds, 'refexp')
        from models.postprocessors import build_postprocessors
        postprocessors = build_postprocessors(args, dataset_meta['val'][0]['dataset_mode'])
        
    else:
        if args.distributed:
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
            if not args.eval:
                sampler_train = DistributedSampler(dataset_train)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            if not args.eval:
                sampler_train = torch.utils.data.RandomSampler(dataset_train)

        collate_fn = utils.collate_fn
        
        if not args.eval:
            batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train, args.batch_size, drop_last=True)
            data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                        collate_fn=collate_fn, num_workers=args.num_workers)

        if dataset_meta['val'][0]['dataset_mode'] in ['d3', 'omni']:
            # When the validation set is omnilabel, force batch size to 1
            args.use_coco_eval = False
            args.batch_size = 1
            
        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                    drop_last=False, collate_fn=collate_fn, num_workers=args.num_workers)
        
    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.epochs, pct_start=0.2)
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(clean_state_dict(checkpoint['model']),strict=False)

    output_dir = Path(args.output_dir)
    
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')) and args.resume is None:
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')

    if os.path.exists(os.path.join(args.output_dir, 'checkpoint_fusion.pth')) and args.fusion_resume is None:
        args.fusion_resume = os.path.join(args.output_dir, 'checkpoint_fusion.pth')
    
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint_projector.pth')) and args.projector_resume is None:
        args.projector_resume = os.path.join(args.output_dir, 'checkpoint_projector.pth')

    if os.path.exists(os.path.join(args.output_dir, 'checkpoint_llm.pth')) and args.llm_resume is None:
        args.llm_resume = os.path.join(args.output_dir, 'checkpoint_llm.pth')
    
    
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            #ipdb.set_trace()
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(clean_state_dict(checkpoint['model']),strict=False)
        
        print(f"{ORG}DINO model weights loaded from trained checkpoint{RESET}")

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
            except:
                ValueError("Optimizer and lr_scheduler not loaded properly")
            if args.start_epoch == 0:
                args.start_epoch = checkpoint['epoch'] + 1

    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        print(f"{ORG}Loaded DINO pre-trained weights{RESET}")
        logger.info(str(_load_output))

    ###### load fusion model ######
    if args.encoder_type == "fusion" and args.fusion_resume is not None:
        print(f"{ORG}Fusion model weights loaded from checkpoint{RESET}")
        fusion_model.load_state_dict(torch.load(args.fusion_resume, map_location='cpu')['model'])
    
    if args.train_projector and args.projector_resume is not None:
        try:
            projector_model.load_state_dict(torch.load(args.projector_resume, map_location='cpu'), strict=True)
        except:
            projector_model.load_state_dict(torch.load(args.projector_resume, map_location='cpu')['model'], strict=True)
        print(f"{ORG}Projector weights loaded from checkpoint{RESET}")

    if args.train_llm and args.llm_resume is not None:
        try:
            llm_model.load_state_dict(torch.load(args.llm_resume, map_location='cpu'), strict=True)
        except:
            llm_model.load_state_dict(torch.load(args.llm_resume, map_location='cpu')['model'], strict=True)
        print(f"{ORG}LLM weights loaded from checkpoint{RESET}")



        

    if args.overwrite:
        assert args.pretrain_model_path is not None, "pretrain_model_path must be provided when overwrite is True"
        # Load the checkpoint from pretrain_model_path
        pretrain_checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')
        pretrain_model_state_dict = clean_state_dict(pretrain_checkpoint['model'])

        # Overwrite layers specified in freeze_keywords
        model_state_dict = model_without_ddp.state_dict()
        for name, param in pretrain_model_state_dict.items():
            for keyword in args.freeze_keywords:
                if keyword in name:
                    if name in model_state_dict:
                        model_state_dict[name].copy_(param)
                        print(f"{RED}Overwrote layer {name} weights{RESET}")
                    else:
                        ValueError(f"Layer {name} not found in current model")
                    break
    
    if args.load_weight_names is not None and not args.eval and not args.resume:
        for weight_name in args.load_weight_names:
            weight_path = os.path.join('./checkpoints', weight_name+'.pth')
            assert os.path.exists(weight_path), f"Not found {weight_path}."
            model_state_dict = model_without_ddp.state_dict()
            weight_state_dict = torch.load(weight_path, map_location='cpu')
            if 'module' in list(weight_state_dict.keys())[0]:
                weight_state_dict = {k[7:]: v for k, v in weight_state_dict.items()}
            for name, param in weight_state_dict.items():
                if name in model_state_dict:
                    model_state_dict[name].copy_(param)
                    print(f"{RED}Overwrote layer {name} weights{RESET}")
                else:
                    ValueError(f"Layer {name} not found in current model")


    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        if dataset_meta['val'][0]['dataset_mode'] == 'refcoco':
            evaluate_refcoco(model=model, 
                                                llm_model=llm_model,
                                                projector_model=projector_model if args.train_projector else None,
                                                tokenizer=tokenizer,
                                                fusion_model=fusion_model if args.encoder_type == "fusion" else None,
                                                hidden_states_layer=args.hidden_states_layer,
                                                criterion = criterion, 
                                                postprocessors = postprocessors,
                                                data_loader=data_loader_val, 
                                                base_ds=base_ds, 
                                                device=device, 
                                                output_dir=args.output_dir, 
                                                wo_class_error=wo_class_error, 
                                                args=args,
                                                image_processor=image_processor,
                                                eval_dataset=dataset_meta['val'][0]['dataset_mode'],
                                                evaluator_list = evaluator_list
                                                )
        else:
            test_stats, coco_evaluator = evaluate(model=model, 
                                                llm_model=llm_model,
                                                projector_model=projector_model if args.train_projector else None,
                                                tokenizer=tokenizer,
                                                fusion_model=fusion_model if args.encoder_type == "fusion" else None,
                                                hidden_states_layer=args.hidden_states_layer,
                                                criterion = criterion, 
                                                postprocessors = postprocessors,
                                                data_loader=data_loader_val, 
                                                base_ds=base_ds, 
                                                device=device, 
                                                output_dir=args.output_dir, 
                                                wo_class_error=wo_class_error, 
                                                args=args,
                                                image_processor=image_processor,
                                                eval_dataset=dataset_meta['val'][0]['dataset_mode']
                                                )
        
        if test_stats is None and coco_evaluator is None:
            print("No results")
            return
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return
    
    print("Start training")

    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=False)

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch  (model=model, 
                                        llm_model=llm_model,
                                        projector_model=projector_model if args.train_projector else None, 
                                        tokenizer=tokenizer,
                                        fusion_model=fusion_model if args.encoder_type == "fusion" else None,
                                        hidden_states_layer=args.hidden_states_layer,
                                        criterion= criterion, 
                                        data_loader=data_loader_train, 
                                        optimizer=optimizer,
                                        device=device,
                                        epoch=epoch,
                                        max_norm= args.clip_max_norm,
                                        wo_class_error=wo_class_error, 
                                        lr_scheduler=lr_scheduler,
                                        args=args,
                                        logger=(logger if args.save_log else None),
                                        output_dir=output_dir,
                                        print_freq=args.print_freq,
                                        image_processor=image_processor,
                                        )
        if args.test_flops:
            break
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']

        if not args.onecyclelr:
            lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if args.encoder_type == "fusion":
                checkpoint_fusion_path = output_dir / 'checkpoint_fusion.pth'
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                if args.encoder_type == "fusion":
                    checkpoint_paths.append(output_dir / f'checkpoint_fusion{epoch:04}.pth')

            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,

                }
                utils.save_on_master(weights, checkpoint_path)
            if args.encoder_type == "fusion":
                for checkpoint_fusion_path in checkpoint_paths:
                    weights_fusion = {
                        'model': fusion_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }
                    utils.save_on_master(weights_fusion, checkpoint_fusion_path)
            
            if args.train_projector:
                checkpoint_projector_path = output_dir / 'checkpoint_projector.pth'
                weights_projector = {
                    'model': projector_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                utils.save_on_master(weights_projector, checkpoint_projector_path)
            if args.train_llm:
                checkpoint_llm_path = output_dir / 'checkpoint_llm.pth'
                weights_llm = {
                    'model': llm_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                utils.save_on_master(weights_llm, checkpoint_llm_path)
                    
        # eval
        test_stats, coco_evaluator = evaluate(model=model, 
                                              llm_model=llm_model,
                                              projector_model=projector_model if args.train_projector else None, 
                                              tokenizer=tokenizer,
                                              fusion_model=fusion_model if args.encoder_type == "fusion" else None,
                                              hidden_states_layer=args.hidden_states_layer,
                                              criterion= criterion, 
                                              postprocessors=postprocessors,
                                              data_loader=data_loader_val, 
                                              base_ds=base_ds,
                                              device=device, 
                                              output_dir=args.output_dir, 
                                              wo_class_error=wo_class_error,
                                              args=args,
                                              logger=(logger if args.save_log else None),
                                              image_processor=image_processor,
                                              eval_dataset=dataset_meta['val'][0]['dataset_mode']
                                              )

        if dataset_meta['val'][0]['dataset_mode'] == 'omni':
            pass
        else:
            map_regular = test_stats['coco_eval_bbox'][0]
            _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
            if _isbest:
                checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
                if args.encoder_type == "fusion":
                    checkpoint_fusion_path = output_dir / 'checkpoint_best_regular_fusion.pth'
                    utils.save_on_master({
                        'model': fusion_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_fusion_path)
                if args.train_projector:
                    checkpoint_projector_path = output_dir / 'checkpoint_best_regular_projector.pth'
                    utils.save_on_master({
                        'model': projector_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_projector_path)

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
            }

            try:
                log_stats.update({'now_time': str(datetime.datetime.now())})
            except:
                pass
            
            epoch_time = time.time() - epoch_start_time
            epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
            log_stats['epoch_time'] = epoch_time_str

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    output_dir / "eval" / name)
            draw_image(args.output_dir)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # remove the copied files.
    copyfilelist = vars(args).get('copyfilelist')
    if copyfilelist and args.local_rank == 0:
        from datasets_dino.data_util import remove
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)

from datasets_dino.coco_eval import CocoEvaluator
from datasets_dino.refexp import RefExpEvaluator
def build_evaluator_list(base_ds, dataset_name):

    """Helper function to build the list of evaluators for a given dataset"""
    evaluator_list = []
    if args.no_detection:
        return evaluator_list
    iou_types = ["bbox"]
    if args.masks:
        iou_types.append("segm")

    evaluator_list.append(CocoEvaluator(base_ds, tuple(iou_types), useCats=False))
    if "refexp" in dataset_name:
        evaluator_list.append(RefExpEvaluator(base_ds, ("bbox")))
    return evaluator_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
