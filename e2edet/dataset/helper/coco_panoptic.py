import os
import math
import json
from io import BytesIO

import torch
import numpy as np
from torchvision.datasets.vision import VisionDataset
from PIL import Image

from e2edet.utils.distributed import get_rank, get_world_size
from e2edet.utils.box_ops import masks_to_boxes


class CocoPanoptic(VisionDataset):
    def __init__(
        self,
        root,
        annFile,
        annFolder=None,
        num_replicas=None,
        rank=None,
        transform=None,
        target_transform=None,
        transforms=None,
        cache_mode=False,
    ):
        super(CocoPanoptic, self).__init__(
            root, transforms, transform, target_transform
        )

        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()

        with open(annFile, "r") as f:
            self.coco = json.load(f)

        # sort 'images' field so that they are aligned with 'annotations'
        # i.e., in alphabetical order
        self.coco["images"] = sorted(self.coco["images"], key=lambda x: x["id"])
        # sanity check
        # if "annotations" in self.coco:
        #     for img, ann in zip(self.coco["images"], self.coco["annotations"]):
        #         assert (
        #             img["file_name"][:-4] == ann["file_name"][:-4]
        #         ), "{} != {}".format(img["file_name"], ann["file_name"])

        self.ids = list(map(lambda x: x["id"], self.coco["images"]))
        self.cache_mode = cache_mode
        self.anno_folder = annFolder
        self.rank = rank
        self.num_replicas = num_replicas
        self.num_samples = int(math.ceil(len(self.ids) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        if cache_mode:
            self.cache = {}
            self.cache_images()

    def cache_images(self):
        indices = torch.arange(len(self.ids)).tolist()
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        offset = self.num_samples * self.rank
        indices = set(indices[offset : offset + self.num_samples])

        self.cache = {}
        for index, img_id in enumerate(self.ids):
            if index not in indices:
                continue

            path = self.coco.loadImgs(img_id)[0]["file_name"]
            with open(os.path.join(self.root, path), "rb") as f:
                self.cache[path] = f.read()

    def get_image(self, path):
        if not os.path.isabs(path):
            if os.path.exists(os.path.join(self.root, path)):
                path = os.path.join(self.root, path)
            else:
                path = os.path.join(
                    self.root, path.split("_")[0], path.replace("_gtFine", "")
                )

        if self.cache_mode:
            if path not in self.cache.keys():
                print("Not found image in the cache")
                with open(path, "rb") as f:
                    self.cache[path] = f.read()

            return Image.open(BytesIO(self.cache[path])).convert("RGB")

        return Image.open(path).convert("RGB")

    def __getitem__(self, index):
        from panopticapi.utils import rgb2id
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned ``coco.loadAnns``,
        """
        ann_info = (
            self.coco["annotations"][index]
            if "annotations" in self.coco
            else self.coco["images"][index]
        )

        img = self.get_image(self.coco["images"][index]["file_name"])
        w, h = img.size

        target = {}
        target["image_id"] = (
            ann_info["image_id"] if "image_id" in ann_info else ann_info["id"]
        )

        target["size"] = torch.as_tensor([int(h), int(w)])
        target["orig_size"] = torch.as_tensor([int(h), int(w)])

        if "segments_info" in ann_info:
            ann_path = os.path.join(self.anno_folder, ann_info["file_name"])
            masks = np.asarray(Image.open(ann_path), dtype=np.uint32)
            masks = rgb2id(masks)

            ids = np.array([ann["id"] for ann in ann_info["segments_info"]])
            masks = masks == ids[:, None, None]

            masks = torch.as_tensor(masks, dtype=torch.bool)
            labels = torch.tensor(
                [ann["category_id"] for ann in ann_info["segments_info"]],
                dtype=torch.int64,
            )

            target["masks"] = masks
            target["labels"] = labels

            target["boxes"] = masks_to_boxes(masks)
            for name in ["iscrowd", "area"]:
                target[name] = torch.tensor(
                    [ann[name] for ann in ann_info["segments_info"]]
                )

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.coco["images"])

    def get_height_and_width(self, idx):
        img_info = self.coco["images"][idx]
        height = img_info["height"]
        width = img_info["width"]

        return height, width
