import os
import math
from io import BytesIO

import torch
import numpy as np
from torchvision.datasets.vision import VisionDataset
from PIL import Image

from e2edet.utils.distributed import get_rank, get_world_size
from e2edet.utils.box_ops import masks_to_boxes


class ADESemantic(VisionDataset):
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
        super(ADESemantic, self).__init__(root, transforms, transform, target_transform)

        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()

        ade = self._get_ade20k_pairs(root, annFolder)
        self.ade = sorted(ade, key=lambda x: x["image_id"])

        self.cache_mode = cache_mode
        self.rank = rank
        self.num_replicas = num_replicas
        self.num_samples = int(math.ceil(len(self.ade) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        if cache_mode:
            self.cache = {}
            self.cache_images()

    def _get_ade20k_pairs(self, imgFolder, annFolder):
        ade_pairs = []

        for filename in os.listdir(imgFolder):
            basename, _ = os.path.splitext(filename)
            img_id = int(basename.split("/")[-1].split("_")[-1])
            if filename.endswith(".jpg"):
                img_path = os.path.join(imgFolder, filename)

                if annFolder is not None:
                    maskname = basename + ".png"
                    mask_path = os.path.join(annFolder, maskname)
                    if os.path.isfile(mask_path):
                        ade_pairs.append(
                            {
                                "image_id": img_id,
                                "image_path": img_path,
                                "mask_path": mask_path,
                            }
                        )
                    else:
                        print("Cannot find mask:", mask_path)
                else:
                    ade_pairs.append(
                        {"image_id": img_id, "image_path": img_path, "mask_path": None}
                    )
        print(f"Found {len(ade_pairs)} images in the folder {imgFolder}")

        return ade_pairs

    def cache_images(self):
        indices = torch.arange(len(self.ade)).tolist()
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        offset = self.num_samples * self.rank
        indices = set(indices[offset : offset + self.num_samples])

        self.cache = {}
        for index, item in enumerate(self.ade):
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
        item = self.ade[index]

        img = self.get_image(item["image_path"])
        w, h = img.size

        target = {}
        target["image_id"] = torch.tensor([item["image_id"]])

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
