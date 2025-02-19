from functools import partial

import numpy as np
import torch
import pycocotools.mask as mask_util
from pycocotools import mask as coco_mask

from e2edet.dataset import BaseDataset, register_task
from e2edet.dataset.helper import CocoDetection, LvisDetection, collate2d
from e2edet.utils.box_ops import (
    box_cxcywh_to_xyxy,
    convert_to_xywh,
    paste_grid,
    mask_process,
)
from e2edet.utils.distributed import synchronize


@register_task("detection")
class COCODetection(BaseDataset):
    def __init__(self, config, dataset_type, imdb_file, **kwargs):
        dataset_name = config.get("dataset_name", "coco")

        super().__init__(
            config,
            dataset_name,
            dataset_type,
            current_device=kwargs["current_device"],
            global_config=kwargs["global_config"],
        )

        cache_mode = config.get("sampler", "standard") == "shard"
        if dataset_name == "coco":
            self.data = CocoDetection(
                self._get_absolute_path(imdb_file["image_folder"]),
                self._get_absolute_path(imdb_file["anno_file"]),
                cache_mode=cache_mode,
            )
            self.max_dets = 100
        elif dataset_name == "lvis":
            self.data = LvisDetection(
                self._get_absolute_path(imdb_file["image_folder"]),
                self._get_absolute_path(imdb_file["anno_file"]),
                cache_mode=cache_mode,
            )
            self.max_dets = 300
        else:
            raise ValueError("Only coco|lvis are supported for dataset_name")

        self.image_size = config["image_size"][dataset_type]
        self.threshold = config.get("threshold", None)

        self.prepare = ConvertCocoPolysToMask(config["use_mask"])
        self.loss_mode = self._global_config.loss.params.matcher.params.loss_mode
        print("loss_mode:", self.loss_mode)

    def get_answer_size(self):
        if hasattr(self, "category_processor"):
            return self.category_processor.get_size()

        return self.answer_processor.get_size()

    def get_api(self):
        if self.dataset_name == "coco":
            return self.data.coco
        elif self.dataset_name == "lvis":
            return self.data.lvis

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {}

        img, target = self.data[idx]
        image_id = self.data.ids[idx]

        if self._dataset_type == "test":
            target = {"image_id": image_id, "annotations": []}
        else:
            target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)

        sample["image"] = img
        sample["image_size"] = self.image_size
        if self._dataset_type == "train":
            sample, target = self.image_train_processor(sample, target)
        else:
            sample, target = self.image_test_processor(sample, target)

        return sample, target

    def get_collate_fn(self):
        return partial(collate2d, iter_per_update=self.iter_per_update)

    def prepare_for_evaluation(self, predictions):
        coco_results = []
        for orig_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = None
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

            coco_results.extend(
                [
                    {
                        "image_id": orig_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                        "segmentation": rles[k] if rles is not None else None,
                    }
                    for k, box in enumerate(boxes)
                ]
            )

        return coco_results

    def _process_output(
        self,
        out_logits,
        indices,
        image_sizes,
        target_sizes,
        boxes,
        ins_scores,
        ins_masks=None,
        return_rles=False,
    ):
        top_boxes = indices.div(out_logits.shape[2], rounding_mode="floor")
        labels = indices % out_logits.shape[2]
        img_h, img_w = image_sizes.unbind(1)

        if boxes is not None:
            boxes = torch.gather(boxes, 1, top_boxes.unsqueeze(-1).repeat(1, 1, 4))
            h, w = target_sizes.unbind(1)
            scale_fct = torch.stack([w, h, w, h], dim=1)
            boxes = boxes * scale_fct[:, None, :]

        masks = None
        if ins_masks is not None:
            top_boxes = top_boxes[:, :, None, None].repeat(
                1, 1, ins_masks.shape[-2], ins_masks.shape[-1]
            )
            ins_masks = torch.gather(ins_masks, 1, top_boxes)

            masks = []
            scores = []
            for i, (cur_mask, cur_size) in enumerate(zip(ins_masks, target_sizes)):
                h, w = cur_size.tolist()
                mh, mw = cur_mask.shape[-2:]

                if (mh == img_h and mw == img_w) or (
                    self.image_size is not None
                    and mh == self.image_size
                    and mw == self.image_size
                ):
                    image_size = image_sizes[i].tolist()
                    mask = mask_process(cur_mask, image_size, h, w).sigmoid()
                elif boxes is not None:
                    mask = paste_grid(cur_mask.float().sigmoid(), boxes[i], (h, w))
                pred_mask = (mask >= 0.5).float()

                mask_scores = (mask * pred_mask).sum(-1).sum(-1) / pred_mask.sum(
                    -1
                ).sum(-1)
                score = ins_scores[i] * mask_scores

                scores.append(score)
                if return_rles:
                    pred_mask = pred_mask.cpu()
                    rles = [
                        mask_util.encode(
                            np.array(m[:, :, np.newaxis], dtype=np.uint8, order="F")
                        )[0]
                        for m in pred_mask
                    ]
                    for polygon in rles:
                        polygon["counts"] = polygon["counts"].decode("utf-8")

                    masks.append(rles)
                else:
                    masks.append(pred_mask.byte())
        else:
            scores = ins_scores

        return labels, boxes, masks, scores

    @torch.no_grad()
    def format_for_evalai(self, output, targets, return_rles=False):
        # """Perform the computation
        # Parameters:
        #     outputs: raw outputs of the model
        #     target_sizes: tensor of dimension [batch_size, 2] containing the size of each images of the batch
        #                   For evaluation, this must be the original image size (before any data augmentation)
        #                   For visualization, this should be the image size after data augment, but before padding
        # """
        out_logits = output["pred_logits"]
        target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        image_sizes = torch.stack([t["size"] for t in targets], dim=0)

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        assert image_sizes.shape[1] == 2

        if self.loss_mode == "ce":
            out_logits = out_logits.softmax(dim=-1)[..., :-1]
        else:
            out_logits = out_logits.sigmoid()
        prob = out_logits.reshape(out_logits.shape[0], -1)

        if "pred_boxes" in output:
            out_bbox = output["pred_boxes"]
            out_boxes = box_cxcywh_to_xyxy(out_bbox)
        else:
            out_boxes = None

        if self.threshold is None:
            topk_values, topk_indexes = torch.topk(
                prob, self.max_dets, dim=1, sorted=False
            )
            scores = topk_values

            if "pred_masks" in output:
                ins_masks = output["pred_masks"]
            else:
                ins_masks = None

            labels, boxes, masks, scores = self._process_output(
                out_logits,
                topk_indexes,
                image_sizes,
                target_sizes,
                out_boxes,
                scores,
                ins_masks=ins_masks,
                return_rles=return_rles,
            )

            results = [{"scores": s, "labels": l} for s, l in zip(scores, labels)]
            for i, result in enumerate(results):
                if masks is not None:
                    if return_rles:
                        result["rles"] = masks[i]
                    else:
                        result["masks"] = masks[i]

                if boxes is not None:
                    result["boxes"] = boxes[i]
        else:
            top_pred = prob > self.threshold
            indices = torch.arange(
                top_pred.shape[1], dtype=torch.int64, device=out_logits.device
            )
            results = []
            for i in range(out_logits.shape[0]):
                if not torch.any(top_pred[i]).item():
                    result = (
                        {"scores": [], "labels": [], "boxes": [], "rles": []}
                        if return_rles
                        else {"scores": [], "labels": [], "boxes": [], "masks": []}
                    )
                    results.append(result)
                    continue

                top_indexes = torch.masked_select(indices, top_pred[i])
                scores = torch.masked_select(prob[i], top_pred[i])

                if "pred_masks" in output:
                    ins_masks = output["pred_masks"][i : i + 1]
                else:
                    ins_masks = None

                labels, boxes, masks, scores = self._process_output(
                    out_logits,
                    top_indexes[None],
                    image_sizes[i : i + 1],
                    target_sizes[i : i + 1],
                    out_boxes[i : i + 1] if out_boxes is not None else None,
                    scores[None],
                    ins_masks=ins_masks,
                    return_rles=return_rles,
                )

                result = {"scores": scores[0], "labels": labels[0]}
                if masks is not None:
                    if return_rles:
                        result["rles"] = masks[0]
                    else:
                        result["masks"] = masks[0]

                if boxes is not None:
                    result["boxes"] = boxes[0]
                results.append(result)

        predictions = {
            target["image_id"].item(): output
            for target, output in zip(targets, results)
        }

        return predictions


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def _process_polygons(self, polygons_per_instance):
        def _make_array(t):
            if isinstance(t, torch.Tensor):
                t = t.cpu().numpy()
            return np.asarray(t).astype("float64")

        if not isinstance(polygons_per_instance, list):
            raise ValueError(
                "Cannot create polygons: Expect a list of polygons per instance. "
                "Got '{}' instead.".format(type(polygons_per_instance))
            )

        polygons_per_instance = [_make_array(p) for p in polygons_per_instance]
        for polygon in polygons_per_instance:
            if len(polygon) % 2 != 0 or len(polygon) < 6:
                raise ValueError(
                    f"Cannot create a polygon from {len(polygon)} coordinates."
                )

        return polygons_per_instance

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks.bool()
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno]
        )
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks
