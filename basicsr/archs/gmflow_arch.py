import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from basicsr.archs.gmflow.gmflow.gmflow import GMFlow


class FlowGenerator(nn.Module):
    """GM flow generation.

    Args:
        path (str): Pre-trained path. Default: None.
        requires_grad (bool): If true, the parameters of VGG network will be
            optimized. Default: False.
    """

    def __init__(self,
                 path=None,
                 requires_grad=False,):
        super().__init__()

        self.model = GMFlow()

        if path != None:
            weights = torch.load(
                path, map_location=lambda storage, loc: storage)['model']
            self.model.load_state_dict(weights, strict=True)

        if not requires_grad:
            self.model.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.model.train()
            for param in self.parameters():
                param.requires_grad = True

    def forward(self, im1, im2,
                attn_splits_list=[2],
                corr_radius_list=[-1],
                prop_radius_list=[-1]):
        """Forward function.

        Args:
            im1 (Tensor): Input tensor with shape (n, c, h, w).
            im2 (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        assert im1.shape == im2.shape
        N, C, H, W = im1.shape

        im1 = (im1 + 1) / 2 * 255
        im2 = (im2 + 1) / 2 * 255

        flow = self.model(im1, im2,
                          attn_splits_list=attn_splits_list,
                          corr_radius_list=corr_radius_list,
                          prop_radius_list=prop_radius_list,
                          pred_bidir_flow=False)['flow_preds'][-1]
        # backward_flow = flow[N:]

        return flow


if __name__ == '__main__':
    h, w = 512, 512
    # model = RAFT().cuda()
    model = FlowGenerator(
        load_path='../../weights/GMFlow/gmflow_sintel-0c07dcb3.pth').cuda()
    model.eval()
    print(model)

    x = torch.randn((1, 3, h, w)).cuda()
    y = torch.randn((1, 3, h, w)).cuda()
    with torch.no_grad():
        out = model(x, y)
    pdb.set_trace()
    print(out.shape)
