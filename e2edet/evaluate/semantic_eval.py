import torch

from e2edet.utils.distributed import is_master


class SemanticEvaluator:
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.predictions = []
        self.results = None
        self.reset()

    def get_pretty_results(self):
        string = f"\n"

        mIoU = self.results["mIoU"]
        pixAcc = self.results["pixAcc"]
        string += f"IoU metric: semantic segmentation \n"
        string += f"\t mIoU = {(mIoU):.2f} \n"
        string += f"\t pixAcc = {(pixAcc * 100):.2f} \n"

        return string

    def update(self, preds, labels):
        """Updates the internal evaluation result.
        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """

        def evaluate_worker(self, pred, label):
            correct, labeled = batch_pix_accuracy(pred, label)
            inter, union = batch_intersection_union(pred, label, self.num_classes)

            self.total_correct += correct
            self.total_label += labeled
            if self.total_inter.device != inter.device:
                self.total_inter = self.total_inter.to(inter.device)
                self.total_union = self.total_union.to(union.device)
            self.total_inter += inter
            self.total_union += union

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, preds, labels)
        elif isinstance(preds, (list, tuple)):
            for (pred, label) in zip(preds, labels):
                evaluate_worker(self, pred, label)

    def summarize(self):
        if is_master():
            eps = 1e-6
            pixAcc = 1.0 * self.total_correct / (self.total_label + eps)
            IoU = 1.0 * self.total_inter / (self.total_union + eps)
            mIoU = IoU.mean().item()

            return pixAcc, mIoU

        return None

    def reset(self):
        self.total_inter = torch.zeros(self.num_classes)
        self.total_union = torch.zeros(self.num_classes)
        self.total_correct = 0
        self.total_label = 0


# pytorch version
def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are numpy array, output 4D, target 3D
    predict = torch.argmax(output, dim=1) + 1
    target = target.long() + 1

    pixel_labeled = torch.sum(target > 0).item()
    pixel_correct = torch.sum((predict == target) * (target > 0)).item()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1
    target = target.float() + 1

    predict = predict.float() * (target > 0).float()
    intersection = predict * (predict == target).float()
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)
    area_union = area_pred + area_lab - area_inter
    assert (
        torch.sum(area_inter > area_union).item() == 0
    ), "Intersection area should be smaller than Union area"
    return area_inter.float(), area_union.float()
