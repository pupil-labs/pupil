"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
# Adapted from https://github.com/amdegroot/ssd.pytorch/blob/master/layers/functions/detection.py


import torch
from torch.autograd import Function
from calibration_routines.fingertip_calibration.models.layers import box_utils


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, cfg):
        super(Detect, self).__init__()
        self.num_classes = cfg["num_classes"]
        self.top_k = cfg["max_num_detection"]
        self.nms_thresh = cfg["nms_thresh"]
        self.confidence_thresh = cfg["confidence_thresh"]
        self.bkg_label = 0
        self.variance = [0.1, 0.2]

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = box_utils.decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(self.num_classes):
                if cl == self.bkg_label:
                    continue
                c_mask = conf_scores[cl].gt(self.confidence_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                res = box_utils.nms(boxes, scores, self.nms_thresh, self.top_k)
                if res is not None:
                    ids, count = res
                    output[i, cl, :count] = torch.cat(
                        (scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1
                    )

        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output
