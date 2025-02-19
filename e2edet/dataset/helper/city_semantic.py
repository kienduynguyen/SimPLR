import os
import math
from io import BytesIO

import torch
import numpy as np
from torchvision.datasets.vision import VisionDataset
from PIL import Image

from e2edet.utils.distributed import get_rank, get_world_size
from e2edet.utils.box_ops import masks_to_boxes


class CitySemantic(VisionDataset):
    def __init__(
        self,
        root,
        annFolder=None,
        num_replicas=None,
        rank=None,
        transform=None,
        target_transform=None,
        transforms=None,
        cache_mode=False,
    ):
        super(CitySemantic, self).__init__(
            root, transforms, transform, target_transform
        )

        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()

        city = self._get_city_pairs(root, annFolder)
        self.city = sorted(city, key=lambda x: x["image_id"])

        self.cache_mode = cache_mode
        self.rank = rank
        self.num_replicas = num_replicas
        self.num_samples = int(math.ceil(len(self.ade) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        if cache_mode:
            self.cache = {}
            self.cache_images()

    def _get_city_pairs(imgFolder, annFolder):
        city_pairs = []

        for root, _, files in os.walk(imgFolder):
            for filename in files:
                if filename.endswith(".png"):
                    img_path = os.path.join(root, filename)
                    if not os.path.isfile(img_path):
                        print("Cannot find image:", img_path)
                        continue

                    if annFolder is not None:
                        foldername = os.path.basename(os.path.dirname(img_path))
                        maskname = filename.replace("leftImg8bit", "gtFine_labelIds")
                        mask_path = os.path.join(annFolder, foldername, maskname)
                        if os.path.isfile(mask_path):
                            city_pairs.append(
                                {"image_path": img_path, "mask_path": mask_path}
                            )
                        else:
                            print("Cannot find mask:", mask_path)
                    else:
                        city_pairs.append({"image_path": img_path, "mask_path": None})
        print(f"Found {len(city_pairs)} images in the folder {imgFolder}")

        return city_pairs

    def cache_images(self):
        indices = torch.arange(len(self.city)).tolist()
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        offset = self.num_samples * self.rank
        indices = set(indices[offset : offset + self.num_samples])

        self.cache = {}
        for index, item in enumerate(self.city):
            img_path = item["image_path"]
            if index not in indices:
                continue

            with open(img_path, "rb") as f:
                self.cache[img_path] = f.read()

    def get_image(self, path):
        if self.cache_mode:
            if path not in self.cache.keys():
                print("Not found image in the cache")
                with open(os.path.join(self.root, path), "rb") as f:
                    self.cache[path] = f.read()

            return Image.open(BytesIO(self.cache[path])).convert("RGB")

        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned ``coco.loadAnns``,
        """
        item = self.city[index]

        img = self.get_image(item["image_path"])
        w, h = img.size

        target = {}
        target["image_id"] = torch.tensor([index])

        target["size"] = torch.as_tensor([int(h), int(w)])
        target["orig_size"] = torch.as_tensor([int(h), int(w)])

        if item["mask_path"] is not None:
            masks = np.asarray(Image.open(item["mask_path"]), dtype=np.uint32)
            labels = np.unique(masks)

            masks = masks == labels[:, None, None]
            masks = torch.as_tensor(masks, dtype=torch.bool)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            target["masks"] = masks
            target["labels"] = labels

            target["boxes"] = masks_to_boxes(masks)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ade)
