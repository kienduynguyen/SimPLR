import os
import json
import collections
import warnings

import torch

from e2edet.evaluate import (
    CocoEvaluator,
    CocoProposalEvaluator,
    PanopticEvaluator,
    LvisEvaluator,
)
from e2edet.utils.distributed import (
    is_dist_avail_and_initialized,
    get_world_size,
    all_gather,
    synchronize,
    is_master,
)
from e2edet.utils.functional import recursive_copy_to_device
from e2edet.dataset.helper import Prefetcher
from e2edet.trainer.engine import BaseEngine, register_engine


@register_engine("detection", "detection3d", "panoptic")
class DetEngine(BaseEngine):
    def __init__(self, trainer):
        super().__init__(trainer)
        self.eval_mode = trainer.running_config.get("eval_mode", "gpu")

    def _compute_loss(self, model_output, target, num_boxes=None):
        if self.model.training:
            assert isinstance(
                model_output, collections.abc.Mapping
            ), "A dict must be returned from the forward of the model."

            if "losses" in model_output:
                warnings.warn(
                    "'losses' already present in model output. "
                    "No calculation will be done in base model."
                )

                assert isinstance(
                    model_output["losses"], collections.abc.Mapping
                ), "'losses' must be a dict."
            else:
                losses_stat = {}
                if num_boxes is not None:
                    model_output["num_boxes"] = num_boxes

                loss_dict = self.trainer.losses(model_output, target)
                if hasattr(self.trainer.losses, "weight_dict"):
                    weight_dict = self.trainer.losses.weight_dict
                    total_loss = sum(
                        loss_dict[k] * weight_dict[k]
                        for k in loss_dict.keys()
                        if k in weight_dict
                    )
                    losses_stat.update(
                        {f"{k}_unscaled": v for k, v in loss_dict.items()}
                    )
                    losses_stat.update(
                        {
                            k: v * weight_dict[k]
                            for k, v in loss_dict.items()
                            if k in weight_dict
                        }
                    )
                else:
                    total_loss = sum(loss_dict[k] for k in loss_dict.keys())
                    losses_stat.update({k: v for k, v in loss_dict.items()})
                losses_stat["total_loss"] = total_loss
                model_output["losses"] = total_loss
                model_output["losses_stat"] = losses_stat

            if "metrics" in model_output:
                warnings.warn(
                    "'metrics' already present in model output. "
                    "No calculation will be done in base model."
                )

                assert isinstance(
                    model_output["metrics"], collections.abc.Mapping
                ), "'metrics' must be a dict."
            else:
                metrics = {}
                for name, metric in self.trainer.metrics.items():
                    if name == "accuracy":
                        metrics.update(
                            metric(*self.trainer.losses.get_target_classes())
                        )
                    else:
                        metrics.update(metric(model_output, target))
                model_output["metrics"] = metrics

        return model_output

    @torch.no_grad()
    def evaluate(self, split):
        self.trainer.writer.write(f"Evaluation time. Running on full {split} set...")
        self.trainer.timers[split].reset()

        iter_per_update = self.trainer.iter_per_update
        dataset = self.datasets[split]
        dataloader = self.dataloaders[split]

        coco_evaluator = None
        lvis_evaluator = None
        accumulated_results = None
        panoptic_evaluator = None
        other_args = {}
        self.trainer.model_without_ddp.inference()

        if split == "test":
            accumulated_results = {}
        else:
            self.model.eval()
            if "bbox" in self.trainer.iou_type or "segm" in self.trainer.iou_type:
                if dataset.dataset_name == "coco":
                    coco_evaluator = CocoEvaluator(
                        dataset.get_api(), self.trainer.iou_type
                    )
                    other_args["return_rles"] = split == "test"
                elif dataset.dataset_name == "lvis":
                    lvis_evaluator = LvisEvaluator(
                        dataset.get_api(), self.trainer.iou_type
                    )
                    other_args["return_rles"] = True
            if "proposal" in self.trainer.iou_type:
                coco_evaluator = CocoProposalEvaluator(dataset.get_api())
            if "panoptic" in self.trainer.iou_type:
                tmp_folder = self.trainer.running_config.get("panoptic_folder", None)
                if tmp_folder is not None:
                    output_dir = os.path.join(
                        tmp_folder,
                        self.trainer.running_config.save_dir,
                        "panoptic_eval",
                    )
                else:
                    output_dir = os.path.join(
                        self.trainer.running_config.save_dir, "panoptic_eval"
                    )
                panoptic_evaluator = PanopticEvaluator(
                    dataset.anno_file, dataset.anno_folder, output_dir=output_dir
                )

        prefetcher = Prefetcher(
            self.dataloaders[split], self.datasets[split], prefetch=False
        )

        for _ in range(len(dataloader)):
            batch = prefetcher.get_next_sample()

            if iter_per_update > 1:
                results = {}
                for splitted_batch in batch:
                    outputs, targets = self._forward(splitted_batch)

                    if self.eval_mode == "cpu":
                        results = dataset.format_for_evalai(
                            recursive_copy_to_device(
                                outputs, False, torch.device("cpu")
                            ),
                            recursive_copy_to_device(
                                targets, False, torch.device("cpu")
                            ),
                            **other_args,
                        )
                    else:
                        results.update(
                            dataset.format_for_evalai(outputs, targets, **other_args)
                        )
            else:
                outputs, targets = self._forward(batch)

                if self.eval_mode == "cpu":
                    results = dataset.format_for_evalai(
                        recursive_copy_to_device(outputs, False, torch.device("cpu")),
                        recursive_copy_to_device(targets, False, torch.device("cpu")),
                        **other_args,
                    )
                else:
                    results = dataset.format_for_evalai(outputs, targets, **other_args)
            self.trainer.profile("Post-processing time")

            if coco_evaluator is not None:
                coco_evaluator.update(results)

            if lvis_evaluator is not None:
                lvis_evaluator.update(results)

            if panoptic_evaluator is not None:
                panoptic_evaluator.update(
                    recursive_copy_to_device(results, True, torch.device("cpu"))
                )

            if accumulated_results is not None:
                accumulated_results.update(
                    recursive_copy_to_device(results, True, torch.device("cpu"))
                )

        stats = {
            "update": self.trainer.current_update,
            "epoch": self.current_epoch,
            "max_update": self.trainer.max_update,
            "num_image": len(dataset),
        }
        if coco_evaluator is not None:
            coco_evaluator.synchronize_between_processes()
            coco_evaluator.accumulate()
            coco_evaluator.summarize()
            if "bbox" in self.trainer.iou_type:
                stats["coco_eval_bbox"] = coco_evaluator.get_pretty_results("bbox")
            if "segm" in self.trainer.iou_type:
                stats["coco_eval_segm"] = coco_evaluator.get_pretty_results("segm")
            if "proposal" in self.trainer.iou_type:
                stats["coco_eval_proposal"] = coco_evaluator.get_pretty_results()

        if lvis_evaluator is not None:
            lvis_evaluator.synchronize_between_processes()
            lvis_evaluator.summarize()
            if "bbox" in self.trainer.iou_type:
                stats["lvis_eval_bbox"] = lvis_evaluator.get_pretty_results("bbox")
            if "segm" in self.trainer.iou_type:
                stats["lvis_eval_segm"] = lvis_evaluator.get_pretty_results("segm")

        if panoptic_evaluator is not None:
            panoptic_evaluator.synchronize_between_processes()
            panoptic_evaluator.summarize()
            if is_master():
                stats["coco_eval_panoptic"] = panoptic_evaluator.get_pretty_results()

        if accumulated_results is not None:
            accumulated_results = all_gather(accumulated_results)

            if is_master():
                merged_results = {}
                for result in accumulated_results:
                    merged_results.update(result)
                accumulated_results = None

                if self.trainer.iou_type is None:
                    save_file = os.path.join(
                        self.trainer.checkpoint.ckpt_foldername, "results.pth"
                    )
                    torch.save(merged_results, save_file)

                    dataset.prepare_for_evaluation(
                        merged_results, self.trainer.checkpoint.ckpt_foldername
                    )
                else:
                    merged_results = dataset.prepare_for_evaluation(merged_results)
                    test_path = os.path.join(
                        self.trainer.checkpoint.ckpt_foldername, "test_result.json"
                    )
                    json.dump(merged_results, open(test_path, "w"))
        synchronize()

        self.trainer._print_log(split, stats)
        self.trainer._update_tensorboard(split)

        self.model.train()
        self.trainer.timers["train"].reset()

    def train_epoch(self):
        current_epoch = self.trainer.current_epoch
        current_update = self.trainer.current_update
        max_update = self.trainer.max_update
        iter_per_update = self.trainer.iter_per_update
        eval_interval = self.trainer.eval_interval
        save_interval = self.trainer.save_interval

        prefetcher = Prefetcher(
            self.trainer.dataloaders["train"],
            self.trainer.datasets["train"],
            prefetch=False,
        )

        if self.trainer.samplers["train"] is not None and self.trainer.parallel:
            self.trainer.writer.write(f"Seeding with epoch: {current_epoch}")
            self.trainer.samplers["train"].set_epoch(current_epoch)

        for idx in range(len(self.trainer.dataloaders["train"])):
            self.trainer.profile("Batch prepare time")

            batch = prefetcher.get_next_sample()
            self.optimizer.zero_grad(set_to_none=True)

            if iter_per_update > 1:
                num_boxes = 0
                for split in batch:
                    num_boxes += sum(len(t["labels"]) for t in split[1])
                num_boxes = torch.tensor(
                    [num_boxes], dtype=torch.float, device=self.trainer.device
                )
                if is_dist_avail_and_initialized():
                    torch.distributed.all_reduce(num_boxes)
                num_boxes = torch.clamp(num_boxes / get_world_size(), min=1)

                assert iter_per_update == len(batch)
                for splitted_batch in batch:
                    # splitted_batch[0]["num_boxes"] = num_boxes
                    output = self._forward(splitted_batch, num_boxes=num_boxes)[0]
                    if output is None:
                        continue
                    self.trainer._sync_losses_and_metrics("train", output)
                    self._backward(output)
            else:
                output = self._forward(batch)[0]
                if output is None:
                    continue
                self.trainer._sync_losses_and_metrics("train", output)
                self._backward(output)

            current_update = self._step(current_update)
            self.trainer.profile("Optimizer stepping time")

            if current_update == self.trainer.current_update:
                self.trainer.writer.write("Skipping iteration...", "warning")
                continue

            if current_update > max_update:
                break

            self.lr_scheduler.step(current_update)

            assert self.trainer.current_update == (current_update - 1)
            self.trainer.current_update = current_update
            self.trainer.gc_handler.run(current_update)
            self._update_info("train", current_update)

            if current_update % save_interval == 0:
                self.trainer.writer.write("Checkpoint time. Saving a checkpoint...")
                self.trainer.checkpoint.save(current_update)

            if current_update % eval_interval == 0 and "val" in self.trainer.run_type:
                self.evaluate("val")

        self.lr_scheduler.step_epoch(current_epoch)
