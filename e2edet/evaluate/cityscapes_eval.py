import os
import glob
import logging
from PIL import Image

import numpy as np
import torch

from e2edet.utils.distributed import get_rank, synchronize


class CityscapesInstanceEvaluator:
    def __init__(self, gt_folder, output_dir="cityscapes_instance_eval"):
        self.gt_folder = gt_folder
        self.output_dir = output_dir
        self._logger = logging.getLogger(__name__)

    def get_pretty_results(self):
        string = f"\n"
        string += f"IoU metric: instance \n"
        ap = self.results["AP"]
        ap50 = self.results["AP50"]
        string += f"  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {(ap * 100):.2f}\n"
        string += f"  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {(ap50 * 100):.2f}\n"

        return string

    def update(self, predictions):
        from cityscapesscripts.helpers.labels import name2label

        for prediction in predictions:
            file_name = prediction["file_name"]
            basename = os.path.splitext(os.path.basename(file_name))[0]
            pred_txt = os.path.join(self.output_dir, basename + "_pred.txt")

            masks = prediction["masks"].cpu()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            num_instances = len(labels)
            with open(pred_txt, "w") as fout:
                for i in range(num_instances):
                    pred_class = labels[i]
                    class_id = name2label[pred_class].id
                    score = scores[i]
                    mask = masks[i].numpy().astype("uint8")
                    png_filename = os.path.join(
                        self.output_dir, basename + "_{}_{}.png".format(i, pred_class)
                    )

                    Image.fromarray(mask * 255).save(png_filename)
                    fout.write(
                        "{} {} {}\n".format(
                            os.path.basename(png_filename), class_id, score
                        )
                    )

    def summarize(self):
        synchronize()
        if get_rank() > 0:
            return

        import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as cityscapes_eval

        self._logger.info("Evaluating results under {} ...".format(self.output_dir))

        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = os.path.abspath(self.output_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False
        cityscapes_eval.args.gtInstancesFile = os.path.join(
            self.output_dir, "gtInstances.json"
        )

        gt_folder = self.gt_folder
        groundTruthImgList = glob.glob(
            os.path.join(gt_folder, "*", "*_gtFine_instanceIds.png")
        )
        assert len(
            groundTruthImgList
        ), "Cannot find any ground truth images to use for evaluation. Searched for {}".format(
            cityscapes_eval.args.groundTruthSearch
        )

        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(
                cityscapes_eval.getPrediction(gt, cityscapes_eval.args)
            )
        results = cityscapes_eval.evaluateImgLists(
            predictionImgList, groundTruthImgList, cityscapes_eval.args
        )["averages"]

        self.results = {"AP": results["allAp"], "AP50": results["allAp50%"]}


class CityscapesSemSegEvaluator:
    def __init__(self, gt_folder, output_dir="cityscapes_instance_eval"):
        self.gt_folder = gt_folder
        self.output_dir = output_dir
        self._logger = logging.getLogger(__name__)

    def get_pretty_results(self):
        string = f"\n"
        string += f"IoU metric: instance \n"
        iou = self.results["IoU"]
        iiou = self.results["iIoU"]
        iou_sup = self.results["IoU_sup"]
        iiou_sup = self.results["iIoU_sup"]
        string += f"  IoU       = {(iou * 100):.2f}\n"
        string += f"  iIoU      = {(iiou * 100):.2f}\n"
        string += f"  IoU_sup   = {(iou_sup * 100):.2f}\n"
        string += f"  iIoU_sup  = {(iiou_sup * 100):.2f}\n"

        return string

    def update(self, predictions):
        from cityscapesscripts.helpers.labels import trainId2label

        for prediction in predictions:
            file_name = prediction["file_name"]
            basename = os.path.splitext(os.path.basename(file_name))[0]
            pred_filename = os.path.join(self.output_dir, basename + "_pred.png")

            output = prediction["segm_seg"].argmax(dim=0).cpu().numpy()
            pred = 255 * np.ones(output.shape, dtype=np.uint8)
            for train_id, label in trainId2label.items():
                if label.ignoreInEval:
                    continue
                pred[output == train_id] = label.id
            Image.fromarray(pred).save(pred_filename)

    def summarize(self):
        synchronize()
        if get_rank() > 0:
            return

        import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as cityscapes_eval

        self._logger.info("Evaluating results under {} ...".format(self.output_dir))

        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = os.path.abspath(self.output_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False

        gt_folder = self.gt_folder
        groundTruthImgList = glob.glob(
            os.path.join(gt_folder, "*", "*_gtFine_labelIds.png")
        )
        assert len(
            groundTruthImgList
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            cityscapes_eval.args.groundTruthSearch
        )
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(
                cityscapes_eval.getPrediction(cityscapes_eval.args, gt)
            )
        results = cityscapes_eval.evaluateImgLists(
            predictionImgList, groundTruthImgList, cityscapes_eval.args
        )

        self.results = {
            "IoU": results["averageScoreClasses"],
            "iIoU": results["averageScoreInstClasses"],
            "IoU_sup": results["averageScoreCategories"],
            "iIoU_sup": results["averageScoreInstCategories"],
        }
