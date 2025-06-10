# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import io
from functools import partial

import torch
import torch.nn.functional as F
from PIL import Image

from e2edet.dataset import BaseDataset, register_task
from e2edet.dataset.helper import CocoPanoptic, collate2d
from e2edet.utils.box_ops import mask_process, mask_process_w_boxes, box_cxcywh_to_xyxy


@register_task("panoptic")
class COCOPanoptic(BaseDataset):
    def __init__(self, config, dataset_type, imdb_file, **kwargs):
        if "name" in kwargs:
            dataset_name = kwargs["name"]
        elif "dataset_name" in kwargs:
            dataset_name = kwargs["dataset_name"]
        else:
            dataset_name = "coco"

        super().__init__(
            config,
            dataset_name,
            dataset_type,
            current_device=kwargs["current_device"],
            global_config=kwargs["global_config"],
        )

        cache_mode = config.get("sampler", "standard") == "shard"
        if dataset_type == "test":
            self.coco_dataset = CocoPanoptic(
                self._get_absolute_path(imdb_file["image_folder"]),
                self._get_absolute_path(imdb_file["anno_file"]),
                cache_mode=cache_mode,
            )
        else:
            self.coco_dataset = CocoPanoptic(
                self._get_absolute_path(imdb_file["image_folder"]),
                self._get_absolute_path(imdb_file["anno_file"]),
                self._get_absolute_path(imdb_file["anno_folder"]),
                cache_mode=cache_mode,
            )
            self.anno_folder = self._get_absolute_path(imdb_file["anno_folder"])
        self.anno_file = self._get_absolute_path(imdb_file["anno_file"])
        self.image_size = config["image_size"][dataset_type]

        self.object_mask_threshold = config.object_mask_threshold
        self.overlap_threshold = config.overlap_threshold
        self.focal_label = self._global_config.model_config[
            self._global_config.model
        ].get("focal_label", True)
        print("setting focal_label:", self.focal_label)

    def get_answer_size(self):
        if hasattr(self, "category_processor"):
            return self.category_processor.get_size()

        return self.answer_processor.get_size()

    def get_model_params(self):
        return {"num_classes": self.answer_processor.get_size()}

    def get_api(self):
        return self.coco_dataset.coco

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, idx):
        sample = {}

        img, target = self.coco_dataset[idx]

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
        return predictions

    @torch.no_grad()
    def format_for_evalai(self, output, targets):
        from panopticapi.utils import id2rgb

        # """Perform the computation
        # Parameters:
        #     outputs: raw outputs of the model
        #     target_sizes: tensor of dimension [batch_size, 2] containing the size of each images of the batch
        #                   For evaluation, this must be the original image size (before any data augmentation)
        #                   For visualization, this should be the image size after data augment, but before padding
        # """
        # pano_temp = 0.06

        out_logits, out_masks = output["pred_logits"], output["pred_masks"]
        target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        image_sizes = torch.stack([t["size"] for t in targets], dim=0)

        # out_boxes = box_cxcywh_to_xyxy(output["pred_boxes"])
        # img_h, img_w = target_sizes.unbind(1)
        # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        # out_boxes = out_boxes * scale_fct[:, None, :]

        assert len(out_logits) == len(target_sizes)

        results = []

        for i, (out_logit, out_mask, image_size, target_size) in enumerate(
            zip(out_logits, out_masks, image_sizes, target_sizes)
        ):
            image_size = image_size.tolist()
            h, w = target_size.tolist()

            out_mask = mask_process(out_mask, image_size, h, w)
            # out_mask = mask_process_w_boxes(out_mask, out_boxes[i], image_size, h, w)
            out_mask = out_mask.sigmoid()

            if not self.focal_label:
                prob = out_logit.softmax(dim=-1)
            else:
                prob = out_logit.sigmoid()

            scores, labels = prob.max(-1)
            # scores, labels = F.softmax(prob / pano_temp, dim=-1).max(-1)
            num_classes = int(self.answer_processor.get_size())
            keep = labels.ne(num_classes) & (scores > self.object_mask_threshold)

            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = out_mask[keep]

            cur_prob_masks = cur_scores.unsqueeze(-1).unsqueeze(-1) * cur_masks

            panoptic_seg = torch.zeros(
                (h, w), dtype=torch.int32, device=cur_masks.device
            )
            segments_info = []

            current_segment_id = 0

            if cur_masks.shape[0] > 0:
                cur_mask_ids = cur_prob_masks.argmax(0)
                stuff_memory_list = {}

                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = self.answer_processor.is_thing_ids(pred_class)
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.overlap_threshold:
                            continue

                        # merge stuff regions
                        if not isthing:
                            if int(pred_class) in stuff_memory_list.keys():
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                stuff_memory_list[int(pred_class)] = (
                                    current_segment_id + 1
                                )

                        current_segment_id += 1
                        panoptic_seg[mask] = current_segment_id

                        segments_info.append(
                            {
                                "id": current_segment_id,
                                "isthing": bool(isthing),
                                "category_id": int(pred_class),
                            }
                        )
            panoptic_seg = Image.fromarray(id2rgb(panoptic_seg.cpu().numpy()))

            with io.BytesIO() as out:
                panoptic_seg.save(out, format="PNG")
                item = {
                    "png_string": out.getvalue(),
                    "segments_info": segments_info,
                }
            results.append(item)

        predictions = {
            target["image_id"]: output for target, output in zip(targets, results)
        }

        return predictions
