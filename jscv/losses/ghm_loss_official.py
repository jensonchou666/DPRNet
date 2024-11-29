import torch
import torch.nn as nn
import torch.nn.functional as F



# def _expand_onehot_labels(labels, label_weights, label_channels):

#     bin_labels = labels.new_full((labels.size(0), label_channels), 0)
#     inds = torch.nonzero(
#         (labels >= 0) & (labels < label_channels), as_tuple=False).squeeze()
#     if inds.numel() > 0:
#         bin_labels[inds, labels[inds]] = 1
#     bin_label_weights = label_weights.view(-1, 1).expand(
#         label_weights.size(0), label_channels)
#     return bin_labels, bin_label_weights

def _expand_onehot_labels(labels, label_weights, label_channels):

    lb = labels.clone()
    lb[~label_weights] = 0
    lb = F.one_hot(lb, num_classes=label_channels)

    label_weights = label_weights.unsqueeze(-1).expand_as(lb)

    return lb, label_weights



# TODO: code refactoring to make it consistent with other losses

class GHMC(nn.Module):
    """GHM Classification Loss.

    Details of the theorem can be viewed in the paper
    `Gradient Harmonized Single-stage Detector
    <https://arxiv.org/abs/1811.05181>`_.

    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """

    def __init__(self, bins=100, momentum=0, ignore_index=None, use_sigmoid=True, loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins  # 应该论文中epsilon=2/bin
        self.momentum = momentum
        edges = torch.arange(bins + 1).float() / bins  # g纵坐标均匀切割范围
        self.register_buffer('edges', edges)
        self.edges[-1] += 1e-6
        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer('acc_sum', acc_sum)
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index


    def forward(self, pred, target, label_weight=None, *args, **kwargs):
        """Calculate the GHM-C loss.

        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        #TODO
        assert pred.dim() != target.dim()

        if label_weight is None:
            label_weight = torch.ones_like(target)
            if self.ignore_index is not None:
                label_weight[target == self.ignore_index] = 0

        # the target should be binary class label

        if pred.dim() != target.dim():
            target, label_weight = _expand_onehot_labels(target, label_weight, pred.size(1))
        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        pred_bhwc = pred.permute(0, 2, 3, 1)

        weights = torch.zeros_like(pred_bhwc)

        # gradient length g函数
        # g: [B, C]
        g = torch.abs(pred_bhwc.sigmoid().detach() - target)

        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            # 计算在指定edges范围内的样本个数
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            # 如果某个区间范围内没有样本，则直接忽略
            if num_in_bin > 0:
                if mmt > 0:
                    # ema操作
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin  # tot=论文中N，num_in_bin=论文中delta
                n += 1
        if n > 0:
            # weights=论文中beta 每个样本的权重
            weights = weights / n  # n是有样本的区间个数即论文中的1/l(l是单位间隔)，n<=bin
        loss = F.binary_cross_entropy_with_logits(
            pred_bhwc, target, weights, reduction='sum') / tot
        return loss * self.loss_weight