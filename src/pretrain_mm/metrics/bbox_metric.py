import torch
import torchmetrics

from pretrain_mm.utils.eval_utils import box_distance_fn


class BBoxDistance(torchmetrics.Metric):
    """computes the L2 distance between two bounding boxes
    inputs will look like
    prediction: <box>100, 200, 300, 400</box>
    and target: <box>101, 201, 301, 401</box>
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("seen", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("parsed", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("min_dist", default=torch.tensor(torch.inf), dist_reduce_fx="min")
        self.add_state("max_dist", default=torch.tensor(-torch.inf), dist_reduce_fx="max")

        self.box_open_tag = "<box>"
        self.box_close_tag = "</box>"

        self._dist_func = self._distance_func

    def _distance_func_torch(self, preds: str, target: str) -> torch.Tensor:
        pred_boxes = torch.tensor([float(x) for x in preds.strip(self.box_open_tag + self.box_close_tag).split(",")])
        target_boxes = torch.tensor([float(x) for x in target.strip(self.box_open_tag + self.box_close_tag).split(",")])
        return torch.norm(pred_boxes - target_boxes, p=2)

    def _distance_func(self, preds: str, target: str) -> torch.Tensor:
        if (distance := box_distance_fn(preds, target)) is not None:
            distance = torch.tensor(distance)
        return distance

    def update(self, preds: str, target: str) -> None:
        self.seen += 1
        # Compute the L2 distance between the predicted bounding box and the target bounding box
        # distance either
        if (distance := self._dist_func(preds, target)) is not None:
            # Update the metric states
            self.distance += distance
            self.parsed += 1
            self.min_dist = torch.min(self.min_dist, distance)
            self.max_dist = torch.max(self.max_dist, distance)

    def compute(self):
        if self.parsed == 0:
            return torch.tensor(torch.inf)  # Avoid division by zero

        return {
            "BBox_distance": self.distance / self.parsed,
            "BBox_error_percent": (self.seen - self.parsed) / self.seen,
            "BBox_min": self.min_dist,
            "BBox_max": self.max_dist,
        }
