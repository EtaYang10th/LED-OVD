# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""Postprocessors class to transform MDETR output according to the downstream task"""
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
import ipdb

class PostProcessFlickr(nn.Module):
    """This module converts the model's output for Flickr30k entities evaluation.

    This processor is intended for recall@k evaluation with respect to each phrase in the sentence.
    It requires a description of each phrase (as a binary mask), and returns a sorted list of boxes for each phrase.
    """

    @torch.no_grad()
    def forward(self, outputs, target_sizes, positive_map, items_per_batch_element):
        """Perform the computation.
        Args:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
            positive_map: tensor [total_nbr_phrases x max_seq_len] for each phrase in the batch, contains a binary
                          mask of the tokens that correspond to that sentence. Note that this is a "collapsed" batch,
                          meaning that all the phrases of all the batch elements are stored sequentially.
            items_per_batch_element: list[int] number of phrases corresponding to each batch element.
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        batch_size = target_sizes.shape[0]

        prob = F.softmax(out_logits, -1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        # and from relative [0, 1] to absolute [0, height] coordinates
        boxes = boxes * scale_fct[:, None, :]

        cum_sum = np.cumsum(items_per_batch_element)

        curr_batch_index = 0
        # binarize the map if not already binary
        pos = positive_map > 1e-6

        predicted_boxes = [[] for _ in range(batch_size)]

        # The collapsed batch dimension must match the number of items
        assert len(pos) == cum_sum[-1]

        if len(pos) == 0:
            return predicted_boxes

        # if the first batch elements don't contain elements, skip them.
        while cum_sum[curr_batch_index] == 0:
            curr_batch_index += 1

        for i in range(len(pos)):
            # scores are computed by taking the max over the scores assigned to the positive tokens
            scores, _ = torch.max(pos[i].unsqueeze(0) * prob[curr_batch_index, :, :], dim=-1)
            _, indices = torch.sort(scores, descending=True)

            assert items_per_batch_element[curr_batch_index] > 0
            predicted_boxes[curr_batch_index].append(boxes[curr_batch_index][indices].to("cpu").tolist())
            if i == len(pos) - 1:
                break

            # check if we need to move to the next batch element
            while i >= cum_sum[curr_batch_index] - 1:
                curr_batch_index += 1
                assert curr_batch_index < len(cum_sum)

        return predicted_boxes


class PostProcessPhrasecut(nn.Module):
    """This module converts the model's output for Phrasecut evaluation.

    Contrary to most postprocessors, it doesn't operate directly on model's output, but must be called
    after the regular box processor, and optionally the segmentation post-processor (if masks are being computed).

    Args:
        score_thresh(float): minimum score at which the box/mask will be kept
    """

    def __init__(self, score_thresh=0.7):
        super().__init__()
        self.score_thresh = score_thresh

    @torch.no_grad()
    def forward(self, results):
        """Perform the computation
        Args:
            results: Dictionnary that comes from PostProcess and optionally PostProcessSegm
        """
        final_results = []
        for elem in results:
            keep = elem["scores"] > self.score_thresh
            # convert boxes to [x0, y0, w, h]
            boxes = elem["boxes"][keep].view(-1, 4)
            boxes[..., 2:] -= boxes[..., :2]
            res = {"boxes": boxes.tolist()}
            if "masks" in elem:
                res["masks"] = elem["masks"][keep].any(0).squeeze(0).cpu().numpy()
            final_results.append(res)

        return final_results

def remove_full_stop(description_list):
    ret_list = []
    for descript in description_list:
        if descript[-1] == '.':
            descript = descript[:-1]    # remove '.'
        ret_list.append(descript)
    return ret_list

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the refcoco evaluation"""
    def __init__(self, num_select=50, nms_iou_threshold=-1, args=None) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold

    @torch.no_grad()
    def forward(self, outputs, target_sizes, pos_maps,not_to_xyxy=False, test=False):
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        # import ipdb; ipdb.set_trace()
        # 计算目标分数（文本匹配得分）
        #scores = out_logits.sigmoid().max(dim=-1)[0]  # 取多token中的最大概率值
        prob_to_token = out_logits.sigmoid()
        prob = prob_to_token @ pos_maps.T
        #prob = prob.view(prob.shape[0], -1)
        labels = torch.ones_like(prob.view(prob.shape[0], -1))
        
        # 选择top-k预测

        topk_values, topk_indexes = torch.topk(prob.view(prob.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, prob.shape[2], rounding_mode='trunc')

        # 转换边界框格式
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # 转换到绝对坐标
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} 
                      for s, l, b in zip(scores, labels, boxes)]

        assert len(scores) == len(labels) == len(boxes)
        # 处理参考表达置信度（如果存在）
        if "pred_isfinal" in outputs:
            is_final = outputs["pred_isfinal"].sigmoid().squeeze(-1)
            scores_refexp = scores * is_final
            for res, score_refexp in zip(results, scores_refexp):
                res["scores_refexp"] = score_refexp

        return results


class PostProcessSegm(nn.Module):
    """Similar to PostProcess but for segmentation masks.

    This processor is to be called sequentially after PostProcess.

    Args:
        threshold: threshold that will be applied to binarize the segmentation masks.
    """

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        """Perform the computation
        Parameters:
            results: already pre-processed boxes (output of PostProcess)
            outputs: raw outputs of the model
            orig_target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
            max_target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                              after data augmentation.
        """
        assert len(orig_target_sizes) == len(max_target_sizes)
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        outputs_masks = outputs["pred_masks"].squeeze(2)
        outputs_masks = F.interpolate(outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False)

        # Check if all sizes are the same, in which case we can do the interpolation more efficiently
        min_h, min_w = max_target_sizes.min(0)[0].tolist()
        min_orig_h, min_orig_w = orig_target_sizes.min(0)[0].tolist()
        max_orig_h, max_orig_w = orig_target_sizes.max(0)[0].tolist()
        if min_h == max_h and min_w == max_w and min_orig_h == max_orig_h and min_orig_w == max_orig_w:
            outputs_masks = (
                F.interpolate(outputs_masks, size=(min_orig_h, min_orig_w), mode="bilinear").sigmoid() > self.threshold
            ).cpu()
            for i, cur_mask in enumerate(outputs_masks):
                results[i]["masks"] = cur_mask.unsqueeze(1)
            return results

        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]["masks"] = (
                F.interpolate(results[i]["masks"].float(), size=tuple(tt.tolist()), mode="bilinear").sigmoid()
                > self.threshold
            ).cpu()

        return results


def build_postprocessors(args, dataset_name) -> Dict[str, nn.Module]:
    postprocessors: Dict[str, nn.Module] = {"bbox": PostProcess()}
    if args.masks:
        postprocessors["segm"] = PostProcessSegm()


    return postprocessors
