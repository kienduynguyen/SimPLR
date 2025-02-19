# ------------------------------------------------------------------------
# BoxeR
# Copyright (c) 2022. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import copy

import torch

from e2edet.utils.distributed import all_gather
from e2edet.utils.box_ops import box_iou_detectron, box_xywh_to_xyxy


class CocoProposalEvaluator:

    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }

    area_ranges = [
        [0**2, 1e5**2],  # all
        [0**2, 32**2],  # small
        [32**2, 96**2],  # medium
        [96**2, 1e5**2],  # large
        [96**2, 128**2],  # 96-128
        [128**2, 256**2],  # 128-256
        [256**2, 512**2],  # 256-512
        [512**2, 1e5**2],
    ]  # 512-inf

    def __init__(self, coco_gt, threshold=None, area=None, limit=None):
        coco_gt = copy.deepcopy(coco_gt)
        self.coco = coco_gt

        assert area in self.areas or area is None, "Unknown area range: {}".format(area)
        if area is None:
            self.area = ["all", "small", "medium", "large"]
            self.gt_overlaps = {"all": [], "small": [], "medium": [], "large": []}
            self.num_pos = {"all": 0, "small": 0, "medium": 0, "large": 0}
            self.stats = {"all": None, "small": None, "medium": None, "large": None}
        else:
            self.area = [area]
            self.gt_overlaps = {area: []}
            self.num_pos = {area: 0}
            self.stats = {area: None}
        self.thresholds = threshold
        self.limit = limit

    def get_pretty_results(self):
        string = f"\n"

        for area in self.area:
            string += f"Average Recall (AR) @[ IoU=0.50:0.95 | area={area} ] = {self.stats[area]['ar']}\n"

        return string

    def update(self, predictions):
        # TODO: consider to move prepare function to dataset
        results = self.prepare(predictions)

        for area in self.area:
            self.evaluate(results, area)

    def synchronize_between_processes(self):
        for area in self.area:
            all_gt_overlaps = all_gather(self.gt_overlaps[area])
            all_num_pos = all_gather(self.num_pos[area])

            merged_gt_overlaps = []
            for gt_overlaps in all_gt_overlaps:
                merged_gt_overlaps.extend(gt_overlaps)
            self.gt_overlaps[area] = merged_gt_overlaps
            self.num_pos[area] = sum(all_num_pos)

    def accumulate(self):
        for area in self.area:
            gt_overlaps = (
                torch.cat(self.gt_overlaps[area], dim=0)
                if len(self.gt_overlaps[area])
                else torch.zeros(0, dtype=torch.float32)
            )
            gt_overlaps, _ = torch.sort(gt_overlaps)
            self.gt_overlaps[area] = gt_overlaps

    def summarize(self):
        if self.thresholds is None:
            step = 0.05
            self.thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
        recalls = torch.zeros_like(self.thresholds)

        for area in self.area:
            # compute recall for each iou threshold
            for i, t in enumerate(self.thresholds):
                recalls[i] = (self.gt_overlaps[area] >= t).float().sum() / float(
                    self.num_pos[area]
                )
            # ar = 2 * np.trapz(recalls, thresholds)
            ar = recalls.mean()

            self.stats[area] = {
                "ar": ar,
                "recalls": recalls,
                "thresholds": self.thresholds,
                "gt_overlaps": self.gt_overlaps,
                "num_pos": self.num_pos,
            }

    def prepare(self, predictions):
        return predictions

    def evaluate(self, predictions, area="all"):
        area_range = self.area_ranges[self.areas[area]]

        for orig_id, prediction in predictions.items():
            inds = prediction["scores"].sort(descending=True)[1]
            boxes = prediction["boxes"][inds].cpu()

            ann_ids = self.coco.getAnnIds(imgIds=orig_id)
            anno = self.coco.loadAnns(ann_ids)

            gt_boxes = [
                box_xywh_to_xyxy(torch.as_tensor(obj["bbox"]))
                for obj in anno
                if obj["iscrowd"] == 0
            ]

            if len(gt_boxes) == 0 or len(boxes) == 0:
                continue

            gt_boxes = torch.stack(gt_boxes, dim=0).reshape(
                -1, 4
            )  # guard against no boxes
            gt_areas = torch.as_tensor(
                [obj["area"] for obj in anno if obj["iscrowd"] == 0]
            )

            valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
            gt_boxes = gt_boxes[valid_gt_inds]

            self.num_pos[area] += len(gt_boxes)

            if len(gt_boxes) == 0:
                continue

            if self.limit is not None and len(boxes) > self.limit:
                boxes = boxes[: self.limit]

            overlaps = box_iou_detectron(boxes, gt_boxes)

            _gt_overlaps = torch.zeros(len(gt_boxes))
            for j in range(min(len(boxes), len(gt_boxes))):
                # find which proposal box maximally covers each gt box
                # and get the iou amount of coverage for each gt box
                max_overlaps, argmax_overlaps = overlaps.max(dim=0)

                # find which gt box is 'best' covered (i.e. 'best' = most iou)
                gt_ovr, gt_ind = max_overlaps.max(dim=0)
                assert gt_ovr >= 0
                # find the proposal box that covers the best covered gt box
                box_ind = argmax_overlaps[gt_ind]
                # record the iou coverage of this gt box
                _gt_overlaps[j] = overlaps[box_ind, gt_ind]
                assert _gt_overlaps[j] == gt_ovr
                # mark the proposal box and the gt box as used
                overlaps[box_ind, :] = -1
                overlaps[:, gt_ind] = -1

            # append recorded iou coverage level
            self.gt_overlaps[area].append(_gt_overlaps)
