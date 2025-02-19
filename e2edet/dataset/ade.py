from functools import partial

import torch
from PIL import Image

from e2edet.dataset import BaseDataset, register_task
from e2edet.dataset.helper import collate2d, ADESemantic
from e2edet.utils.box_ops import mask_process


@register_task("semantic")
class ADE(BaseDataset):
    def __init__(self, config, dataset_type, imdb_file, **kwargs):
        if "name" in kwargs:
            dataset_name = kwargs["name"]
        elif "dataset_name" in kwargs:
            dataset_name = kwargs["dataset_name"]
        else:
            dataset_name = "ade20k"

        super().__init__(
            config,
            dataset_name,
            dataset_type,
            current_device=kwargs["current_device"],
            global_config=kwargs["global_config"],
        )

        cache_mode = config.get("sampler", "standard") == "shard"
        if dataset_type == "test":
            self.ade_dataset = ADESemantic(
                self._get_absolute_path(imdb_file["image_folder"]),
                cache_mode=cache_mode,
            )
        else:
            self.ade_dataset = ADESemantic(
                self._get_absolute_path(imdb_file["image_folder"]),
                self._get_absolute_path(imdb_file["anno_folder"]),
                cache_mode=cache_mode,
            )
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

    def __len__(self):
        return len(self.ade_dataset)

    def __getitem__(self, idx):
        sample = {}

        img, target = self.ade_dataset[idx]

        sample["image"] = img
        sample["image_size"] = self.image_size
        if self._dataset_type == "train":
            sample, target = self.image_train_processor(sample, target)
        else:
            sample, target = self.image_test_processor(sample, target)

        return sample, target

    def get_collate_fn(self):
        return partial(collate2d, iter_per_update=self.iter_per_update)

    @torch.no_grad()
    def format_for_evalai(self, output, targets):
        # """Perform the computation
        # Parameters:
        #     outputs: raw outputs of the model
        #     target_sizes: tensor of dimension [batch_size, 2] containing the size of each images of the batch
        #                   For evaluation, this must be the original image size (before any data augmentation)
        #                   For visualization, this should be the image size after data augment, but before padding
        # """
        out_logits, out_masks = output["pred_logits"], output["pred_masks"]
        target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        image_sizes = torch.stack([t["size"] for t in targets], dim=0)

        assert len(out_logits) == len(target_sizes)

        results = []

        for i, (out_logit, out_mask, image_size, target_size) in enumerate(
            zip(out_logits, out_masks, image_sizes, target_sizes)
        ):
            image_size = image_size.tolist()
            h, w = target_size.tolist()

            out_mask = mask_process(out_mask, image_size, h, w)
            out_mask = out_mask.sigmoid()

            if not self.focal_label:
                prob = out_logit.softmax(dim=-1)[..., :-1]
            else:
                prob = out_logit.sigmoid()

            semseg = torch.einsum("qc,qhw->chw", prob, out_mask)
            results.append(semseg)

        return results
