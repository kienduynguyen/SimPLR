import os
import math
from io import BytesIO

import torch
from torchvision.datasets.vision import VisionDataset
from PIL import Image

from e2edet.utils.distributed import get_rank, get_world_size


class LvisDetection(VisionDataset):
    def __init__(
        self,
        root,
        annFile,
        num_replicas=None,
        rank=None,
        transform=None,
        target_transform=None,
        transforms=None,
        cache_mode=False,
    ):
        super(LvisDetection, self).__init__(
            root, transforms, transform, target_transform
        )
        from lvis import LVIS

        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()

        self.lvis = LVIS(annFile)
        self.ids = list(sorted(self.lvis.imgs.keys()))
        self.imgs = self.lvis.load_imgs(self.ids)
        self.anns = [self.lvis.img_ann_map[img_id] for img_id in self.ids]
        ann_ids = [ann["id"] for anns_per_image in self.anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(
            ann_ids
        ), "Annotation ids in '{}' are not unique".format(annFile)

        self.cache_mode = cache_mode
        self.rank = rank
        self.num_replicas = num_replicas
        self.num_samples = int(math.ceil(len(self.ids) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self._dataset_dicts = None
        if cache_mode:
            self.cache = {}
            self.cache_images()

    @property
    def dataset_dicts(self):
        if self._dataset_dicts is None:
            self._dataset_dicts = []
            for img_dict, anno_dict_list in zip(self.imgs, self.anns):
                record = {}
                image_id = record["image_id"] = img_dict["id"]

                objs = []
                for anno in anno_dict_list:
                    assert anno["image_id"] == image_id
                    obj = {"category_id": anno["category_id"]}
                    objs.append(obj)
                record["annotations"] = objs
                self._dataset_dicts.append(record)

        return self._dataset_dicts

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

            path = self.lvis.loadImgs(img_id)[0]["file_name"]
            with open(os.path.join(self.root, path), "rb") as f:
                self.cache[path] = f.read()

    def get_image(self, path):
        if self.cache_mode:
            if path not in self.cache.keys():
                print("Not found image in the cache")
                with open(os.path.join(self.root, path), "rb") as f:
                    self.cache[path] = f.read()

            return Image.open(BytesIO(self.cache[path])).convert("RGB")

        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def __getitem__(self, index):
        split_folder, file_name = self.imgs[index]["coco_url"].split("/")[-2:]
        path = os.path.join(split_folder, file_name)

        target = self.anns[index]
        img = self.get_image(path)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)
