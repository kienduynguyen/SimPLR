import os
import copy
import random
import collections.abc as abc
from collections import defaultdict

import numpy as np
import torchvision.transforms as transforms

from e2edet.utils.distributed import is_master
from e2edet.dataset.processor import functional as F
from e2edet.dataset.helper.categories import (
    COCO_CATEGORIES,
    LVIS_CATEGORIES,
    COCO_PANOPTIC_CATEGORIES,
    CITYSCAPES_PANOPTIC_CATEGORIES,
)


PROCESSOR_REGISTRY = {}


def register_processor(name):
    def register_processor_cls(cls):
        if name in PROCESSOR_REGISTRY:
            raise ValueError("Cannot register duplicate processor ({})".format(name))

        PROCESSOR_REGISTRY[name] = cls
        return cls

    return register_processor_cls


def build_processor(config):
    if not hasattr(config, "type"):
        raise AttributeError(
            "Config must have 'type' attribute to specify type of processor"
        )

    if config["type"] in PROCESSOR_REGISTRY:
        processor_class = PROCESSOR_REGISTRY[config["type"]]
    else:
        raise ValueError("Unknown processor type {}".format(config["type"]))

    params = {}
    if not hasattr(config, "params") and is_master():
        print(
            "Config doesn't have 'params' attribute to "
            "specify parameters of the processor "
            "of type {}. Setting to default \{\}".format(config["type"])
        )
    else:
        params = config["params"]

    try:
        processor_instance = processor_class(**params)
    except Exception as e:
        print("Error in", processor_class.__name__)
        raise e

    return processor_instance


class BaseProcessor:
    def __init__(self, params={}):
        for kk, vv in params.items():
            setattr(self, kk, vv)

    def __call__(self, item, *args, **kwargs):
        return item


@register_processor("category")
class CategoryProcessor(BaseProcessor):
    def __init__(self, dataset_name, **kwargs):
        defaults = dict(dataset_name=dataset_name)
        super().__init__(defaults)

        if dataset_name == "lvis":
            self.categories = LVIS_CATEGORIES
        elif dataset_name == "coco":
            self.categories = COCO_CATEGORIES
        elif dataset_name == "coco_panoptic":
            self.categories = COCO_PANOPTIC_CATEGORIES
        elif dataset_name == "cityscapes_panoptic":
            self.categories = CITYSCAPES_PANOPTIC_CATEGORIES
        else:
            raise ValueError(f"Unsupported dataset_name {dataset_name}")
        self.num_cat = len(self.categories)
        self.max_id = max(cat["id"] for cat in self.categories)

        idx2cat, cat2idx = self._process_categories()
        self.idx2cat = idx2cat
        self.cat2idx = cat2idx

    def _process_categories(self):
        categories = ["N/A" for _ in range(self.max_id + 1)]
        cat2idx = defaultdict(int)
        for cat in self.categories:
            categories[cat["id"]] = cat["name"]
            cat2idx[cat["name"]] = cat["id"]

        return categories, cat2idx

    def idx2cat(self, idx):
        return self.idx2cat[idx]

    def cat2idx(self, cat):
        return self.cat2idx[cat]

    def get_size(self):
        return len(self.idx2cat)

    def __len__(self):
        return len(self.idx2cat)


@register_processor("answer")
class AnswerProcessor(BaseProcessor):
    def __init__(self, class_file, data_root_dir=None):
        defaults = dict(class_file=class_file, data_root_dir=data_root_dir)
        super().__init__(defaults)
        if not os.path.isabs(class_file) and data_root_dir is not None:
            class_file = os.path.join(data_root_dir, class_file)

        if not os.path.exists(class_file):
            raise RuntimeError(
                "Vocab file {} for vocab dict doesn't exist!".format(class_file)
            )

        self.word_list = self._load_str_list(class_file)
        self.word2idx_dict = {word: idx for idx, word in enumerate(self.word_list)}

    def _load_str_list(self, class_file):
        with open(class_file) as f:
            lines = f.readlines()
        lines = [self._process_answer(l) for l in lines]

        return lines

    def _process_answer(self, answer):
        remove = [",", "?"]
        answer = answer.lower()

        for item in remove:
            answer = answer.replace(item, "")
        answer = answer.replace("'s", " 's")

        return answer.strip()

    def get_size(self):
        return len(self.word_list)

    def idx2cls(self, n_w):
        return self.word_list[n_w]

    def cls2idx(self, w):
        if w in self.word2idx_dict:
            return self.word2idx_dict[w]
        else:
            raise ValueError("class %s not in dictionary" % w)

    def __len__(self):
        return len(self.word_list)


@register_processor("panoptic_answer")
class PanopticAnswerProcessor(AnswerProcessor):
    def __init__(self, class_file, is_thing_idx, data_root_dir=None):
        super().__init__(class_file, data_root_dir=data_root_dir)

        self.is_thing_idx = is_thing_idx

    def is_thing_ids(self, class_id):
        return class_id <= self.is_thing_idx


@register_processor("cityscapes_panoptic_answer")
class CityScapesPanopticAnswerProcessor(AnswerProcessor):
    def __init__(self, class_file, is_thing_idx, data_root_dir=None):
        super().__init__(class_file, data_root_dir=data_root_dir)

        self.is_thing_idx = is_thing_idx

    def is_thing_ids(self, class_id):
        return class_id >= self.is_thing_idx


# =========================== #
# --------- 2d ops ---------- #
# =========================== #


@register_processor("to_tensor")
class ToTensor(BaseProcessor):
    def __init__(self):
        super().__init__()

    def __call__(self, sample, target=None):
        sample, target = F.to_tensor(sample, target)

        return sample, target


@register_processor("normalize")
class Normalize(BaseProcessor):
    def __init__(self, mean, std, norm=True):
        defaults = dict(mean=mean, std=std, norm=norm)
        super().__init__(defaults)

    def __call__(self, sample, target=None):
        sample, target = F.normalize(
            sample, target, mean=self.mean, std=self.std, norm=self.norm
        )

        return sample, target


@register_processor("random_size_crop")
class RandomSizeCrop(BaseProcessor):
    def __init__(self, min_size, max_size, recompute_boxes=False):
        defaults = dict(
            min_size=min_size, max_size=max_size, recompute_boxes=recompute_boxes
        )
        super().__init__(defaults)

    def __call__(self, sample, target=None):
        img = sample["image"]
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = transforms.RandomCrop.get_params(img, [h, w])

        return F.crop(sample, target, region, recompute_boxes=self.recompute_boxes)


@register_processor("random_resize")
class RandomResize(BaseProcessor):
    def __init__(self, min_size, max_size=None, mode="range"):
        if isinstance(min_size, int):
            min_size = (min_size,)
        else:
            if mode == "range":
                min_size = list(range(*min_size))
            elif mode == "list":
                min_size = min_size
            elif mode == "eval":
                min_size = eval(min_size)
            else:
                raise ValueError("only mode range|list|eval are accepted")
        defaults = dict(min_size=min_size, max_size=max_size)
        super().__init__(defaults)

    def __call__(self, sample, target=None):
        size = random.choice(self.min_size)
        sample, target = F.resize(sample, target, size, self.max_size)

        return sample, target


@register_processor("random_horizontal_flip")
class RandomHorizontalFlip(BaseProcessor):
    def __init__(self, prob=0.5):
        super().__init__(dict(p=prob))

    def __call__(self, sample, target=None):
        if random.random() < self.p:
            sample, target = F.hflip(sample, target)

        return sample, target


@register_processor("random_select")
class RandomSelect(BaseProcessor):
    def __init__(self, preprocessors, probs):
        super().__init__(dict(preprocessors=preprocessors, p=probs))
        self.preprocessors = []
        for preprocessor in preprocessors:
            self.preprocessors.append(build_processor(preprocessor))
        assert len(self.preprocessors) == len(self.p)

    def __call__(self, sample, target=None):
        idx = random.choices(list(range(len(self.preprocessors))), weights=self.p)[0]
        sample, target = self.preprocessors[idx](sample, target)

        return sample, target


@register_processor("resize_scale")
class ResizeScale(BaseProcessor):
    def __init__(self, min_scale, max_scale, image_size):
        super().__init__(
            dict(min_scale=min_scale, max_scale=max_scale, image_size=image_size)
        )

    def __call__(self, sample, target=None):
        scale = random.uniform(self.min_scale, self.max_scale)

        return F.resize_scale(sample, target, scale, self.image_size, self.image_size)


@register_processor("fixed_size_crop")
class FixedSizeCrop(BaseProcessor):
    def __init__(self, image_size, pad_value=128, recompute_boxes=False):
        if isinstance(image_size, int):
            crop_size = (image_size, image_size)
        else:
            crop_size = image_size
        super().__init__(
            dict(
                image_size=image_size,
                pad_value=pad_value,
                crop_size=crop_size,
                recompute_boxes=recompute_boxes,
            )
        )

    def __call__(self, sample, target=None):
        return F.random_crop(
            sample,
            target,
            self.crop_size,
            pad=True,
            pad_value=self.pad_value,
            recompute_boxes=self.recompute_boxes,
        )


@register_processor("random_size_crop_v2")
class RandomSizeCropv2(BaseProcessor):
    def __init__(self, image_size, recompute_boxes=False):
        if isinstance(image_size, int):
            crop_size = (image_size, image_size)
        else:
            crop_size = image_size
        super().__init__(
            dict(
                image_size=image_size,
                crop_size=crop_size,
                recompute_boxes=recompute_boxes,
            )
        )

    def __call__(self, sample, target=None):
        return F.random_crop(
            sample,
            target,
            self.crop_size,
            pad=False,
            recompute_boxes=self.recompute_boxes,
        )


@register_processor("color_aug")
class ColorAug(BaseProcessor):
    def __init__(
        self,
        brightness_delta=32,
        contrast_low=0.5,
        contrast_high=1.5,
        saturation_low=0.5,
        saturation_high=1.5,
        hue_delta=18,
    ):
        super().__init__(
            dict(
                brightness_delta=brightness_delta,
                contrast_low=contrast_low,
                contrast_high=contrast_high,
                saturation_low=saturation_low,
                saturation_high=saturation_high,
                hue_delta=hue_delta,
            )
        )

    def __call__(self, sample, target=None):
        return F.color_aug(
            sample,
            target,
            brightness_delta=self.brightness_delta,
            contrast_low=self.contrast_low,
            contrast_high=self.contrast_high,
            saturation_low=self.saturation_low,
            saturation_high=self.saturation_high,
            hue_delta=self.hue_delta,
        )


# =========================== #
# --------- 3d ops ---------- #
# =========================== #


@register_processor("double_flip")
class DoubleFlip(BaseProcessor):
    def __init__(self):
        super().__init__()

    def __call__(self, sample, target=None):
        sample, target = F.double_flip(sample, target)

        return sample, target


@register_processor("global_rotate")
class GlobalRotation(BaseProcessor):
    def __init__(self, rotation):
        if not isinstance(rotation, abc.Sequence):
            rotation = (-rotation, rotation)
        super().__init__(dict(rotation=rotation))

    def __call__(self, sample, target=None):
        noise_rotation = np.random.uniform(self.rotation[0], self.rotation[1])
        sample, target = F.global_rotation(sample, target, noise_rotation)

        return sample, target


@register_processor("global_scale")
class GlobalScaling(BaseProcessor):
    def __init__(self, min_scale=0.95, max_scale=1.05):
        super().__init__(dict(min_scale=min_scale, max_scale=max_scale))

    def __call__(self, sample, target=None):
        noise_scale = np.random.uniform(self.min_scale, self.max_scale)
        sample, target = F.global_scaling(sample, target, noise_scale)

        return sample, target


@register_processor("global_translate")
class GlobalTranslation(BaseProcessor):
    def __init__(self, noise_translate_std):
        if not isinstance(noise_translate_std, abc.Sequence):
            noise_translate_std = (
                noise_translate_std,
                noise_translate_std,
                noise_translate_std,
            )
        super().__init__(dict(noise_translate_std=noise_translate_std))

    def __call__(self, sample, target=None):
        if (
            self.noise_translate_std[0] == 0
            and self.noise_translate_std[1] == 0
            and self.noise_translate_std[2] == 0
        ):
            return sample, target

        noise_translate = np.array(
            [
                np.random.normal(0, self.noise_translate_std[0], 1),
                np.random.normal(0, self.noise_translate_std[1], 1),
                np.random.normal(0, self.noise_translate_std[2], 1),
            ]
        )[:, 0]
        sample, target = F.global_translate(sample, target, noise_translate)

        return sample, target


@register_processor("shuffle_points")
class ShufflePoints(BaseProcessor):
    def __init__(self):
        super().__init__()

    def __call__(self, sample, target):
        sample, target = F.shuffle_points(sample, target)

        return sample, target


@register_processor("voxelize")
class Voxelize(BaseProcessor):
    def __init__(self, voxel_size, pc_range, max_points_per_voxel, max_voxel_num):
        defaults = dict(
            voxel_size=voxel_size,
            pc_range=pc_range,
            max_points_per_voxel=max_points_per_voxel,
            max_voxel_num=max_voxel_num,
        )
        super().__init__(defaults)

    def __call__(self, sample, target=None):
        sample, target = F.voxelize(
            sample,
            target,
            self.voxel_size,
            self.pc_range,
            self.max_points_per_voxel,
            self.max_voxel_num,
        )

        return sample, target


@register_processor("filter_by_range")
class FilterByRange(BaseProcessor):
    def __init__(self, pc_range):
        pc_range = np.array(list(pc_range))
        super().__init__(dict(pc_range=pc_range))

    def __call__(self, sample, target=None):
        sample, target = F.filter_by_pc_range(sample, target, self.pc_range)

        return sample, target


@register_processor("random_flip")
class RandomFlip(BaseProcessor):
    def __init__(self):
        super().__init__()

    def __call__(self, sample, target=None):
        x_flip = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
        y_flip = np.random.choice([False, True], replace=False, p=[0.5, 0.5])

        sample, target = F.random_flip(sample, target, x_flip, y_flip)

        return sample, target


@register_processor("normalize3d")
class Normalize3d(BaseProcessor):
    def __init__(self, pc_range, normalize_angle="sigmoid"):
        pc_range = np.array(list(pc_range))
        super().__init__(dict(pc_range=pc_range, normalize_angle=normalize_angle))

    def __call__(self, sample, target=None):
        sample, target = F.normalize3d(
            sample, target, self.pc_range, self.normalize_angle
        )

        return sample, target


@register_processor("np_to_tensor")
class NumpyToTensor(BaseProcessor):
    def __init__(self):
        super().__init__()

    def __call__(self, sample, target=None):
        sample, target = F.np_to_tensor(sample, target)

        return sample, target


@register_processor("compose")
class Compose(BaseProcessor):
    def __init__(self, preprocessors):
        super().__init__(dict(preprocessors=preprocessors))
        self.preprocessors = []
        for preprocessor in preprocessors:
            self.preprocessors.append(build_processor(preprocessor))

    def __call__(self, sample, target=None):
        for transform in self.preprocessors:
            sample, target = transform(sample, target)

        return sample, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for transform in self.preprocessors:
            format_string += "\n"
            format_string += "\t{0}".format(transform)
        format_string += "\n"

        return format_string
