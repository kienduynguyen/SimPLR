import json
import os
import contextlib

try:
    from panopticapi.evaluation import pq_compute
except ImportError:
    pass

from e2edet.utils.distributed import all_gather, is_master
from e2edet.utils.general import blockPrint, enablePrint


class PanopticEvaluator:
    def __init__(self, ann_file, ann_folder, output_dir="panoptic_eval"):
        self.gt_json = ann_file
        self.gt_folder = ann_folder
        if is_master():
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.predictions = []
        self.results = None

    def get_pretty_results(self):
        string = f"\n"

        pq_all = self.results["All"]["pq"]
        pq_things = self.results["Things"]["pq"]
        pq_stuff = self.results["Stuff"]["pq"]
        string += f"IoU metric: panoptic quality \n"
        string += f"\t All = {(pq_all * 100):.2f} \n"
        string += f"\t Things = {(pq_things * 100):.2f} \n"
        string += f"\t Stuff = {(pq_stuff * 100):.2f} \n"
        # for cat, value in self.results.items():
        #     string += f"IoU metric: {cat} \n"
        #     for sub_cat, sub_value in value.items():
        #         string += f"\t {sub_cat} = {sub_value} \n"

        return string

    def update(self, predictions):
        processed_predictions = []
        for orig_id, prediction in predictions.items():
            if isinstance(orig_id, str):
                file_name = f"{orig_id}.png"
            else:
                file_name = f"{orig_id:012d}.png"
            prediction["image_id"] = orig_id
            prediction["file_name"] = file_name
            with open(os.path.join(self.output_dir, file_name), "wb") as f:
                f.write(prediction.pop("png_string"))
            processed_predictions.append(prediction)

        self.predictions.extend(processed_predictions)

    def synchronize_between_processes(self):
        all_predictions = all_gather(self.predictions)
        merged_predictions = []
        for p in all_predictions:
            merged_predictions.extend(p)
        self.predictions = merged_predictions

    def summarize(self):
        if is_master():
            json_data = {"annotations": self.predictions}
            predictions_json = os.path.join(self.output_dir, "predictions.json")
            with open(predictions_json, "w") as f:
                f.write(json.dumps(json_data))

            self.predictions = []
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    results = pq_compute(
                        self.gt_json,
                        predictions_json,
                        gt_folder=self.gt_folder,
                        pred_folder=self.output_dir,
                    )
            self.results = results
        return None
