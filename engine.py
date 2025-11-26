"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable

from util.utils import to_device
import torch

import util.misc as utils
from datasets_dino.coco_eval import CocoEvaluator
from datasets_dino.cocogrounding_eval import CocoGroundingEvaluator

from datasets_dino.panoptic_eval import PanopticEvaluator

import ipdb
from preprocess.preprocessing import get_hidden_states
import numpy as np
import matplotlib.pyplot as plt

def train_one_epoch(
                    model: torch.nn.Module=None, 
                    llm_model=None,
                    projector_model = None, 
                    tokenizer=None,
                    fusion_model=None, 
                    hidden_states_layer = -1,
                    criterion: torch.nn.Module = None,
                    data_loader: Iterable = None, 
                    optimizer: torch.optim.Optimizer = None,
                    device: torch.device = torch.device("cuda"),
                    epoch: int = 0, 
                    max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, output_dir=None, print_freq=10, image_processor=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = print_freq
    _cnt = 0

    total_step = 0
    for samples, targets  in metric_logger.log_every(data_loader, print_freq, header, logger=logger, output_dir=output_dir):
        """
        samples:
        images_llm, images_llm_mask= samples.decompose()
            images_llm: [B, C, H, W]
            images_llm_mask: [B, H, W]
        targets: list.
            size: image size,
            boxes: list, [N, 4],
            labels: list, [N],
            ----
            caption: list, [pos+neg],
            cap_list: list, [bs] concatenate all captions separated by ".".
            category: list, stores positive captions.
            label: list, [bs] stores label indices in cap_list for positive captions.
        """
        total_step += 1
        samples = samples.to(device)
        # categorys =[]
        # for t in targets:
        #     category_list = t["category"]
        #     category = ""
        #     for l in range(len(category_list)):
        #         if l != len(category_list)-1:
        #             category = category+ category_list[l] + ', '
        #         else:
        #             category = category+ category_list[l] + '.'
        #     categorys.append(category)

        # print('size:', targets[0]['size'])
        # print('boxes:', targets[0]['boxes'])
        # print('labels:', targets[0]['labels'])
        # print('caption:', targets[0]['caption'])
        # print('cap_list:', targets[0]['cap_list'])
        # print('path:', targets[0]['path'])


        captions = [t["caption"] for t in targets]
        cap_list = [t["cap_list"] for t in targets]
        sample_ids = [t["sample_id"] for t in targets] 
        targets = [{k: v.to(device) for k, v in t.items() if torch.is_tensor(v)} for t in targets]
        images_hidden_states, position_embedding, masks, hw, text_embeds= get_hidden_states(
                                                                                llm_model=llm_model,
                                                                                projector_model=projector_model,
                                                                                tokenizer=tokenizer, 
                                                                                hidden_states_layer=hidden_states_layer, 
                                                                                samples=samples,
                                                                                captions=captions,
                                                                                sample_ids=sample_ids,
                                                                                args=args,
                                                                                image_processor=image_processor)

        if args.encoder_type == "fusion":
            # [B, H*W, C] -> [B, H, W, C]
            with torch.cuda.amp.autocast(enabled=args.amp):
                images_hidden_states[0] = images_hidden_states[0].reshape(images_hidden_states[0].shape[0],hw[0][0],hw[0][1],images_hidden_states[0].shape[-1])
                images_hidden_states = fusion_model(images_hidden_states[0], text_embeds[0],text_embeds[1])
                # [B, C, H, W] -> [B, H, W, C]
                images_hidden_states = images_hidden_states.permute(0,2,3,1)

                images_hidden_states = [images_hidden_states.reshape(images_hidden_states.shape[0],-1,images_hidden_states.shape[-1])]

        """
        outputs: dict.
            'pred_logits': [B, num_queries, C],
            'pred_boxes': [B, num_queries, 4], 
            'text_mask': [B, C],
            'aux_outputs', 
            'token', 
            'interm_outputs', 
            'interm_outputs_for_matching_pre'
        """

        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(samples, captions=captions, images_hidden_states=images_hidden_states, masks=masks, position_embedding=position_embedding, hw=hw, text_embeds=text_embeds)
            if args.test_flops:
                from calflops import calculate_flops
                wrapper_kwargs = {
                    "samples": samples, "captions": captions, "images_hidden_states": images_hidden_states, "masks": masks, "position_embedding": position_embedding, "hw": hw, "text_embeds": text_embeds}
                print("DINO FLOPs: ")
                flops, macs, params = calculate_flops(model=model, kwargs=wrapper_kwargs, forward_mode='costum', include_backPropagation=False, 
                                                      compute_bp_factor=2.0, print_results=True,print_detailed=True,output_as_string=True,output_precision=4, output_unit=None)   
                #print(f"FLOPs: {flops}, MACs: {macs}, Params: {params}")
                return None
            
            loss_dict = criterion(outputs, targets, cap_list, captions)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        check_wight = False
        if check_wight:
            print("mlp1 weight: ")
            print(projector_model.mlp1[1].weight)
            print("mlp1 lr: ")
            for param_group in optimizer.param_groups:
                print(param_group['lr'])

            print("mlp1 weight grad: ")
            print(projector_model.mlp1[1].weight.grad)
            try:
                layer0 = llm_model.language_model.model.layers[0]  # access first layer
            except:
                layer0 = llm_model.module.language_model.model.layers[0]  # access first layer
            q_proj_weight = layer0.self_attn.q_proj.weight    # access q_proj weights
            print("llm_weight: ")
            print(q_proj_weight)
            print("llm_weight grad: ")
            print(q_proj_weight.grad)


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                if args.train_projector:
                    torch.nn.utils.clip_grad_norm_(projector_model.parameters(), max_norm)
                    if args.train_llm:
                        torch.nn.utils.clip_grad_norm_(llm_model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


        if args.save_steps is not None:
            if total_step % args.save_steps == 0:
                if utils.is_main_process():
                    checkpoint_path = os.path.join(output_dir, f'checkpoint_step{total_step}.pth')
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
                    if projector_model is not None:
                        torch.save(projector_model.state_dict(), os.path.join(output_dir, f'checkpoint_projector_step{total_step}.pth'))
                    print(f"Model saved at {checkpoint_path}")
                    if args.train_llm:
                        torch.save(llm_model.state_dict(), os.path.join(output_dir, f'checkpoint_llm_step{total_step}.pth'))
                        print(f"LLM saved at {checkpoint_path}")
                        
                    
                    with open(os.path.join(output_dir, 'step_save.txt'), 'a') as f:
                        f.write(f"{total_step},{loss_value},{optimizer.param_groups[0]['lr']}\n")

                    steps, losses, lrs = [], [], []

                    # Read training info and convert to numeric types
                    with open(os.path.join(output_dir, 'step_save.txt'), 'r') as f:
                        for line in f:
                            step, loss, lr = line.strip().split(',')
                            steps.append(int(step))  # Convert step to int
                            losses.append(float(loss))  # Convert loss to float
                            lrs.append(float(lr))  # Convert learning rate to float

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

                
    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat



@torch.no_grad()
def evaluate(model=None,
                llm_model=None,
                projector_model = None,
                tokenizer=None,
                fusion_model=None,
                hidden_states_layer=-1,
                criterion=None,
                postprocessors=None,
                data_loader=None,
                base_ds=None, 
                device= torch.device("cuda"), 
                output_dir=None, 
                wo_class_error=False, args=None, logger=None, image_processor=None, eval_dataset=None):

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    
    if args.use_coco_eval:
        iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
        useCats = True
        try:
            useCats = args.useCats
        except:
            useCats = True
        if not useCats:
            print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
        
        coco_evaluator = CocoGroundingEvaluator(base_ds, iou_types, useCats=useCats)

        panoptic_evaluator = None
        if 'panoptic' in postprocessors.keys():
            panoptic_evaluator = PanopticEvaluator(
                data_loader.dataset.ann_file,
                data_loader.dataset.ann_folder,
                output_dir=os.path.join(output_dir, "panoptic_eval"),
            )

    _cnt = 0
    output_state_dict = {} # for debug only

    if args.use_coco_eval:
        from pycocotools.coco import COCO
        coco = COCO(args.coco_val_path)

        # Get all categories
        category_dict = coco.loadCats(coco.getCatIds())
        cat_list = [item['name'] for item in category_dict]
        caption = " . ".join(cat_list) + ' .'
        print("Input text prompt:", caption)
    else:
        try:
            cat_list=args.label_list
            caption = " . ".join(cat_list) + ' .'
            print("Input text prompt:", caption)
        except:
                print("Custom dataset loading")

    if eval_dataset is None or eval_dataset == 'coco':
        for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
            samples = samples.to(device)
            targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
            captions = [caption] * len(targets)
            images_hidden_states, position_embedding, masks, hw, text_embeds= get_hidden_states(llm_model=llm_model, 
                                                                                projector_model=projector_model,
                                                                                tokenizer=tokenizer, 
                                                                                hidden_states_layer=hidden_states_layer, 
                                                                                samples=samples,
                                                                                captions=captions,
                                                                                sample_ids=None,
                                                                                args=args,
                                                                                image_processor=image_processor)
            
            if args.encoder_type == "fusion":
                # [B, H*W, C] -> [B, H, W, C]
                with torch.cuda.amp.autocast(enabled=args.amp):
                    images_hidden_states[0] = images_hidden_states[0].reshape(images_hidden_states[0].shape[0],hw[0][0],hw[0][1],images_hidden_states[0].shape[-1])
                    images_hidden_states = fusion_model(images_hidden_states[0], text_embeds[0],text_embeds[1])
                    # [B, C, H, W] -> [B, H, W, C]
                    images_hidden_states = images_hidden_states.permute(0,2,3,1)

                    images_hidden_states = [images_hidden_states.reshape(images_hidden_states.shape[0],-1,images_hidden_states.shape[-1])]

            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = model(samples, captions=captions, images_hidden_states=images_hidden_states, masks=masks, position_embedding=position_embedding, hw=hw, text_embeds=text_embeds)
            
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            # [scores: [100], labels: [100], boxes: [100, 4]] x B
            if 'segm' in postprocessors.keys():
                target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
                
            res = {target['image_id'].item(): output for target, output in zip(targets, results)}

            if coco_evaluator is not None:
                coco_evaluator.update(res)

            if panoptic_evaluator is not None:
                res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
                for i, target in enumerate(targets):
                    image_id = target["image_id"].item()
                    file_name = f"{image_id:012d}.png"
                    res_pano[i]["image_id"] = image_id
                    res_pano[i]["file_name"] = file_name

                panoptic_evaluator.update(res_pano)
            
            if args.save_results:

                for i, (tgt, res) in enumerate(zip(targets, results)):
                    """
                    pred vars:
                        K: number of bbox pred
                        score: Tensor(K),
                        label: list(len: K),
                        bbox: Tensor(K, 4)
                        idx: list(len: K)
                    tgt: dict.

                    """
                    # compare gt and res (after postprocess)
                    gt_bbox = tgt['boxes']
                    gt_label = tgt['labels']
                    gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)

                    _res_bbox = res['boxes']
                    _res_prob = res['scores']
                    _res_label = res['labels']
                    res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
        

                    if 'gt_info' not in output_state_dict:
                        output_state_dict['gt_info'] = []
                    output_state_dict['gt_info'].append(gt_info.cpu())

                    if 'res_info' not in output_state_dict:
                        output_state_dict['res_info'] = []
                    output_state_dict['res_info'].append(res_info.cpu())

                # # for debug only
                # import random
                # if random.random() > 0.7:
                #     print("Now let's break")
                #     break

            _cnt += 1
            if args.debug:
                if _cnt % 15 == 0:
                    print("BREAK!"*5)
                    break

        if args.save_results:
            import os.path as osp
            
            # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
            # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
            savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
            print("Saving res to {}".format(savepath))
            torch.save(output_state_dict, savepath)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        if coco_evaluator is not None:
            coco_evaluator.synchronize_between_processes()
        if panoptic_evaluator is not None:
            panoptic_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        if coco_evaluator is not None:
            coco_evaluator.accumulate()
            coco_evaluator.summarize()
            
        panoptic_res = None
        if panoptic_evaluator is not None:
            panoptic_res = panoptic_evaluator.summarize()
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        if coco_evaluator is not None:
            if 'bbox' in postprocessors.keys():
                stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
            if 'segm' in postprocessors.keys():
                stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
        if panoptic_res is not None:
            stats['PQ_all'] = panoptic_res["All"]
            stats['PQ_th'] = panoptic_res["Things"]
            stats['PQ_st'] = panoptic_res["Stuff"]

        return stats, coco_evaluator
    
    elif eval_dataset == 'omni': 
        from eval_omni import evaluate_omni
        evaluate_omni  (data_loader, 
                        model, 
                        llm_model,
                        projector_model if projector_model is not None else None,
                        tokenizer, 
                        hidden_states_layer, 
                        image_processor, 
                        args, 
                        device, 
                        fusion_model)
        return None, None
    elif eval_dataset == 'd3':
        from eval_d3 import evaluate_d3
        evaluate_d3(data_loader, 
                        model, 
                        llm_model,
                        projector_model if projector_model is not None else None,
                        tokenizer, 
                        hidden_states_layer, 
                        image_processor, 
                        args, 
                        device, 
                        fusion_model)
        return None, None

from datasets_dino.coco_eval import CocoEvaluator
from datasets_dino.refexp import RefExpEvaluator
def evaluate_refcoco(model=None,
                llm_model=None,
                projector_model = None,
                tokenizer=None,
                fusion_model=None,
                hidden_states_layer=-1,
                criterion=None,
                postprocessors=None,
                data_loader=None,
                base_ds=None, 
                device= torch.device("cuda"), 
                output_dir=None,
                evaluator_list = None,
                wo_class_error=False, args=None, logger=None, image_processor=None, eval_dataset=None):
    llm_model.eval()
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    
    _cnt = 0
    output_state_dict = {} # for debug only

    print("refcoco evaluation")

    
    for batch_dict in metric_logger.log_every(data_loader, 10, header, logger=logger): 
        """
        batch_dict.keys():
            dict_keys(['samples', 'targets', 'positive_map'])
            batch_dict['samples'][0].decompose()[0].shape:
                torch.Size([bs, 3, h, w])
            batch_dict['samples'][0].decompose()[1].shape:
                torch.Size([bs, h, w])
            batch_dict['targets'][0].keys():
                dict_keys(['boxes', 'labels', 'caption', 'image_id', 'tokens_positive', 'area', 'iscrowd', 'orig_size', 'size', 'positive_map', 'dataset_name', 'original_id'])
                batch_dict['targets'][0]['image_id']: image id
                batch_dict['targets'][0]['labels']: labels id
                batch_dict['targets'][0]['caption']: caption text
                batch_dict['targets'][0]['boxes']: boxes
                batch_dict['targets'][0]['tokens_positive']: tokens_positive
                batch_dict['targets'][0]['area']: area
                batch_dict['targets'][0]['iscrowd']: 0
                batch_dict['targets'][0]['orig_size']: (h, w)
                batch_dict['targets'][0]['size']: (h, w)
                batch_dict['targets'][0]['positive_map']: positive_map
                batch_dict['targets'][0]['dataset_name']: refcoco
                batch_dict['targets'][0]['original_id']: original_id
        """
        samples = batch_dict['samples']
        samples = samples.todevice(device)
        targets = batch_dict['targets']
        captions = [t["caption"]+" ." for t in targets]
        pos_maps = [t["positive_map"] for t in targets][0].to(device)
        
        
        # pos_maps = torch.zeros((1, 256), dtype=torch.float).to(device)
        # caption = captions[0].replace(".", "")
        # words = caption.split()
        # len(words)
        # att = 1.0/len(words)
        # pos_maps[0, 1:1+len(words)] = att
        #ipdb.set_trace()
        #cur_queries = [t["tokens_positive"] for t in targets]
        #description_list = [t["description"] for t in targets]
        # 
        # for j, label in enumerate(batch_dict['targets'][0]['tokens_positive']):
        # pos_maps = [t["positive_map"] for t in targets][0].to(device)
        #     start_idx = captions[j].find(label)
        # ipdb.set_trace()
        images_hidden_states, position_embedding, masks, hw, text_embeds= get_hidden_states(llm_model=llm_model, 
                                                                            projector_model=projector_model,
                                                                            tokenizer=tokenizer, 
                                                                            hidden_states_layer=hidden_states_layer, 
                                                                            samples=samples,
                                                                            captions=captions,
                                                                            sample_ids=None,
                                                                            args=args,
                                                                            image_processor=image_processor)

        if args.encoder_type == "fusion":
            # [B, H*W, C] -> [B, H, W, C]
            with torch.cuda.amp.autocast(enabled=args.amp):
                images_hidden_states[0] = images_hidden_states[0].reshape(images_hidden_states[0].shape[0],hw[0][0],hw[0][1],images_hidden_states[0].shape[-1])
                images_hidden_states = fusion_model(images_hidden_states[0], text_embeds[0],text_embeds[1])
                # [B, C, H, W] -> [B, H, W, C]
                images_hidden_states = images_hidden_states.permute(0,2,3,1)

                images_hidden_states = [images_hidden_states.reshape(images_hidden_states.shape[0],-1,images_hidden_states.shape[-1])]

        with torch.cuda.amp.autocast(enabled=args.amp):
            
            outputs = model(samples, captions=captions, images_hidden_states=images_hidden_states, masks=masks, position_embedding=position_embedding, hw=hw, text_embeds=text_embeds)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)

        results = postprocessors['bbox'](outputs, orig_target_sizes,pos_maps)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        for evaluator in evaluator_list:
            evaluator.update(res)
        # ipdb.set_trace()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    for evaluator in evaluator_list:
        evaluator.synchronize_between_processes()


    refexp_res = None
    flickr_res = None
    phrasecut_res = None
    for evaluator in evaluator_list:
        if isinstance(evaluator, CocoEvaluator):
            evaluator.accumulate()
            evaluator.summarize()

        elif isinstance(evaluator, (RefExpEvaluator)):
            refexp_res = evaluator.summarize()

    # accumulate predictions from all images

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    for evaluator in evaluator_list:
        if isinstance(evaluator, CocoEvaluator):
            if "bbox" in postprocessors.keys():
                stats["coco_eval_bbox"] = evaluator.coco_eval["bbox"].stats.tolist()
            if "segm" in postprocessors.keys():
                stats["coco_eval_masks"] = evaluator.coco_eval["segm"].stats.tolist()

    if refexp_res is not None:
        stats.update(refexp_res)

    if flickr_res is not None:
        stats["flickr"] = flickr_res

    if phrasecut_res is not None:
        stats["phrasecut"] = phrasecut_res