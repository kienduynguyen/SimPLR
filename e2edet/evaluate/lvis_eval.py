import os
import glob
import warnings
import copy
from PIL import Image

import numpy as np
import torch
from lvis import LVISEval, LVISResults, LVIS
import pycocotools.mask as mask_util

from e2edet.utils.distributed import get_rank, synchronize, all_gather, is_master
from e2edet.utils.box_ops import convert_to_xywh


class LvisEvaluator:
    def __init__(self, lvis_gt, iou_types, max_dets_per_image=None):
        assert isinstance(iou_types, (list, tuple))
        lvis_gt = copy.deepcopy(lvis_gt)
        self.lvis_gt = lvis_gt

        self.iou_types = iou_types
        self.results = {iou_type: [] for iou_type in iou_types}
        self.predictions = {iou_type: [] for iou_type in iou_types}
        self._max_dets_per_image = max_dets_per_image

    def reset(self):
        self.predictions = {iou_type: [] for iou_type in self.iou_types}

    def get_pretty_results(self, iou_type):
        string = f"\n"
        string += f"IoU metric: {iou_type} \n"
        if is_master():
            AP, AP50, AP75, APs, APm, APl, APr, APc, APf = self.results[iou_type]
            string += f"  AP   | maxDets=300 ] = {(AP * 100):.2f}\n"
            string += f"  AP50 | maxDets=300 ] = {(AP50 * 100):.2f}\n"
            string += f"  AP75 | maxDets=300 ] = {(AP75 * 100):.2f}\n"
            string += f"  APs  | maxDets=300 ] = {(APs * 100):.2f}\n"
            string += f"  APm  | maxDets=300 ] = {(APm * 100):.2f}\n"
            string += f"  APl  | maxDets=300 ] = {(APl * 100):.2f}\n"
            string += f"  APr  | maxDets=300 ] = {(APr * 100):.2f}\n"
            string += f"  APc  | maxDets=300 ] = {(APc * 100):.2f}\n"
            string += f"  APf  | maxDets=300 ] = {(APf * 100):.2f}\n"
        else:
            string += "Only master node performs evaluation!"

        return string

    def update(self, predictions):
        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            self.predictions[iou_type].extend(results)

    def synchronize_between_processes(self):
        merged_predictions = {iou_type: [] for iou_type in self.iou_types}
        for iou_type in self.iou_types:
            all_predictions = all_gather(self.predictions[iou_type])
            for p in all_predictions:
                merged_predictions[iou_type].extend(p)
        self.predictions = merged_predictions

    def summarize(self):
        if is_master():
            for iou_type in self.iou_types:
                self.results[iou_type] = _evaluate_predictions_on_lvis(
                    self.lvis_gt,
                    self.predictions[iou_type],
                    iou_type,
                    max_dets_per_image=self._max_dets_per_image,
                )

        self.reset()

        return None

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_lvis_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_lvis_segmentation(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_lvis_detection(self, predictions):
        lvis_results = []
        for orig_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            lvis_results.extend(
                [
                    {
                        "image_id": orig_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )

        return lvis_results

    def prepare_for_lvis_segmentation(self, predictions):
        lvis_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            if "rles" in prediction:
                rles = prediction["rles"]
            elif "masks" in prediction:
                masks = prediction["masks"].cpu()
                rles = [
                    mask_util.encode(
                        np.array(m[:, :, np.newaxis], dtype=np.uint8, order="F")
                    )[0]
                    for m in masks
                ]
                for polygon in rles:
                    polygon["counts"] = polygon["counts"].decode("utf-8")

            lvis_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return lvis_results


def _evaluate_predictions_on_lvis(
    lvis_gt, lvis_results, iou_type, max_dets_per_image=None
):
    """
    Args:
        iou_type (str):
        max_dets_per_image (None or int): limit on maximum detections per image in evaluating AP
            This limit, by default of the LVIS dataset, is 300.
        class_names (None or list[str]): if provided, will use it to predict
            per-category AP.
    Returns:
        a dict of {metric name: score}
    """
    metrics = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"],
    }[iou_type]

    if len(lvis_results) == 0:  # TODO: check if needed
        warnings.warn("No predictions from the model!")
        return {metric: float("nan") for metric in metrics}

    if iou_type == "segm":
        lvis_results = copy.deepcopy(lvis_results)
        # When evaluating mask AP, if the results contain bbox, LVIS API will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in lvis_results:
            c.pop("bbox", None)

    if max_dets_per_image is None:
        max_dets_per_image = 300  # Default for LVIS dataset

    from lvis import LVISEval, LVISResults

    print(f"Evaluating with max detections per image = {max_dets_per_image}")
    lvis_results = LVISResults(lvis_gt, lvis_results, max_dets=max_dets_per_image)
    lvis_eval = LVISEval(lvis_gt, lvis_results, iou_type)
    lvis_eval.run()
    lvis_eval.print_results()

    # Pull the standard metrics from the LVIS results
    results = lvis_eval.get_results()
    # results = {metric: float(results[metric] * 100) for metric in metrics}
    results = [float(results[metric]) for metric in metrics]

    return results
