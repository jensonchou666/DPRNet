import torch
import torch.nn as nn
import kornia
import matplotlib.pyplot as plt

class WeightedBoundaryLoss(nn.Module):
    def __init__(self, edge_weight: float = 10.0, non_edge_weight: float = 1.0):
        """
        带权重的边界损失。

        Args:
            edge_weight (float): 边界像素的权重。
            non_edge_weight (float): 非边界像素的权重。
        """
        super(WeightedBoundaryLoss, self).__init__()
        self.edge_weight = edge_weight
        self.non_edge_weight = non_edge_weight
        self.l1_loss = nn.L1Loss(reduction='none')  # 不进行全局平均，以便加权

    def forward(self, boundary_pred, mask):
        """
        计算加权边界损失。

        Args:
            boundary_pred (torch.Tensor): 模型预测，形状为 [B, 1, H, W]。
            mask (torch.Tensor): 真实标签，形状为 [B, H, W]，值为0或1（二分类）。

        Returns:
            torch.Tensor: 加权边界损失值。
        """
        
        boundary_pred = torch.sigmoid(boundary_pred)
        mask = torch.unsqueeze(mask, 1).float()

        # Sobel 检测边界 (真实边界目标)
        edges = kornia.filters.sobel(mask)  # 直接输出形状 [B, 1, H, W]

        # 生成权重图
        weights = torch.where(edges > 0, self.edge_weight, self.non_edge_weight)  # 边界区域用高权重
        # print(edges.device)
        # 计算加权损失
        loss = self.l1_loss(boundary_pred, edges.to(boundary_pred.device))  # [B, 1, H, W]
        weighted_loss = (loss * weights).mean()  # 加权损失平均
        return weighted_loss