import torch
import numpy as np
from torch import nn
from torch.nn import functional as _nnf
from torchvision.ops import batched_nms, clip_boxes_to_image
from typing import Union, Sequence, Tuple, Optional
from pathlib import Path


class EvalScope(torch.no_grad):
    def __init__(self, model: torch.nn.Module):
        """ Sugar coat that combines `with torch.no_grad` and `model.eval()`.

        Args:
            model: model to be evaluate within the scope
        """
        super(EvalScope, self).__init__()
        self._model = model

    def __enter__(self):
        super(EvalScope, self).__enter__()
        self._prev = self._model.training
        self._model.eval()

    def __exit__(self, exc_type, exc_val, exc_tb):
        super(EvalScope, self).__exit__(exc_type, exc_val, exc_tb)
        self._model.train(self._prev)


class PNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.cv1 = nn.Conv2d(3, 10, kernel_size=(3, 3))
        self.pl1 = nn.PReLU(10)
        self.mp1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.cv2 = nn.Conv2d(10, 16, kernel_size=(3, 3))
        self.pl2 = nn.PReLU(16)
        self.cv3 = nn.Conv2d(16, 32, kernel_size=(3, 3))
        self.pl3 = nn.PReLU(32)
        self.cv4_1 = nn.Conv2d(32, 2, kernel_size=(1, 1))
        self.sm4_1 = nn.Softmax(dim=1)
        self.cv4_2 = nn.Conv2d(32, 4, kernel_size=(1, 1))

        if pretrained:
            state_dict_path = Path(__file__).parent.joinpath('data/pnet.pt')
            self.load_state_dict(torch.load(state_dict_path))

    def forward(self, x):
        x = self.pl1(self.cv1(x))
        x = self.mp1(x)
        x = self.pl2(self.cv2(x))
        x = self.pl3(self.cv3(x))

        a = self.cv4_1(x)
        a = self.sm4_1(a)
        b = self.cv4_2(x)

        return b, a


class RNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.cv1 = nn.Conv2d(3, 28, kernel_size=(3, 3))
        self.pl1 = nn.PReLU(28)
        self.mp1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.cv2 = nn.Conv2d(28, 48, kernel_size=(3, 3))
        self.pl2 = nn.PReLU(48)
        self.mp2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.cv3 = nn.Conv2d(48, 64, kernel_size=(2, 2))
        self.pl3 = nn.PReLU(64)
        self.ds4 = nn.Linear(576, 128)
        self.pl4 = nn.PReLU(128)
        self.ds5_1 = nn.Linear(128, 2)
        self.sm5_1 = nn.Softmax(dim=1)
        self.ds5_2 = nn.Linear(128, 4)

        if pretrained:
            state_dict_path = Path(__file__).parent.joinpath('data/rnet.pt')
            self.load_state_dict(torch.load(state_dict_path))

    def forward(self, x):
        x = self.pl1(self.cv1(x))
        x = self.mp1(x)
        x = self.pl2(self.cv2(x))
        x = self.mp2(x)
        x = self.pl3(self.cv3(x))

        x = x.permute(0, 3, 2, 1).contiguous()

        x = self.ds4(x.view(x.shape[0], -1))
        x = self.pl4(x)
        a = self.ds5_1(x)
        a = self.sm5_1(a)
        b = self.ds5_2(x)

        return b, a


class ONet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.cv1 = nn.Conv2d(3, 32, kernel_size=(3, 3))
        self.pl1 = nn.PReLU(32)
        self.mp1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.cv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.pl2 = nn.PReLU(64)
        self.mp2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.cv3 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        self.pl3 = nn.PReLU(64)
        self.mp3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.cv4 = nn.Conv2d(64, 128, kernel_size=(2, 2))
        self.pl4 = nn.PReLU(128)
        self.ds5 = nn.Linear(1152, 256)
        self.pl5 = nn.PReLU(256)
        self.ds6_1 = nn.Linear(256, 2)
        self.sm6_1 = nn.Softmax(dim=1)
        self.ds6_2 = nn.Linear(256, 4)
        self.ds6_3 = nn.Linear(256, 10)

        if pretrained:
            state_dict_path = Path(__file__).parent.joinpath('data/onet.pt')
            self.load_state_dict(torch.load(state_dict_path))

    def forward(self, x):
        x = self.pl1(self.cv1(x))
        x = self.mp1(x)
        x = self.pl2(self.cv2(x))
        x = self.mp2(x)
        x = self.pl3(self.cv3(x))
        x = self.mp3(x)
        x = self.pl4(self.cv4(x))

        x = x.permute(0, 3, 2, 1).contiguous()

        x = self.ds5(x.view(x.shape[0], -1))
        x = self.pl5(x)
        a = self.ds6_1(x)
        a = self.sm6_1(a)
        b = self.ds6_2(x)
        c = self.ds6_3(x)

        return b, c, a


class MTCNN:
    def __init__(self,
                 threshold: Sequence[float] = (0.6, 0.7, 0.7),
                 device: torch.device = torch.device('cpu'),
                 factor: float = 0.709,
                 minsize: int = 20,
                 nms_threshold: float = 0.5):
        self.pNet = PNet().to(device)
        self.rNet = RNet().to(device)
        self.oNet = ONet().to(device)
        self.pNetThreshold = threshold[0]
        self.rNetThreshold = threshold[1]
        self.oNetThreshold = threshold[2]
        self.factor = factor
        self.device = device
        self.minSize = minsize
        self.nmsThreshold = nms_threshold

    @staticmethod
    def _gather_rois(imgs: torch.Tensor, bbs: torch.Tensor, idxs: torch.Tensor, size: int) -> torch.Tensor:
        _imgs = []
        for j in range(len(idxs)):
            x1, y1, x2, y2 = bbs[j]
            _img = imgs[idxs[j], :, y1:y2, x1:x2].unsqueeze(0)
            _img = _nnf.interpolate(_img, size=(size, size), mode='area')
            _imgs.append(_img)

        _imgs = torch.cat(_imgs)
        return _imgs

    @staticmethod
    def _bb_reg(bbs: torch.Tensor, reg: torch.Tensor) -> torch.Tensor:
        w = bbs[:, 2] - bbs[:, 0]
        h = bbs[:, 3] - bbs[:, 1]

        x1 = bbs[:, 0] + w * reg[:, 0]
        y1 = bbs[:, 1] + h * reg[:, 1]
        x2 = bbs[:, 2] + w * reg[:, 2]
        y2 = bbs[:, 3] + h * reg[:, 3]

        return torch.dstack([x1, y1, x2, y2]).squeeze(0)

    def _first_stage(self, imgs: torch.Tensor):
        with EvalScope(self.pNet):
            _, c, h, w = imgs.shape

            scale = 12.0 / self.minSize  # This is initial scale
            min_l = min(h, w)

            b, s, i = [], [], []

            while min_l * scale >= 12.:
                imgs = _nnf.interpolate(imgs, size=[int(h * scale), int(w * scale)], mode='area')
                reg, pro = self.pNet(imgs)

                pro = pro[:, 1]

                strd = 2. / scale
                cell = 12. / scale

                msk = torch.ge(pro, self.pNetThreshold)  # b, h, w

                if msk.any():
                    indices = msk.nonzero()  # n, 3 <- (i, y, x)
                    idx, r, c = indices[:, 0], indices[:, 1], indices[:, 2]
                    pro = pro[msk]

                    reg = reg.permute(0, 2, 3, 1)  # b, h, w, c <- (x1^, y1^, x2^, y2^)
                    reg = reg[msk]

                    x1, y1 = c * strd, r * strd
                    x2, y2 = x1 + cell, y1 + cell

                    bbs = torch.dstack([x1, y1, x2, y2]).squeeze(0)
                    bbs = self._bb_reg(bbs, reg)
                    nms_idx = batched_nms(bbs, pro, idx, self.nmsThreshold)

                    b.append(bbs[nms_idx])
                    s.append(pro[nms_idx])
                    i.append(idx[nms_idx])

                scale = scale * self.factor

            if len(b) > 0:
                b = torch.cat(b, dim=0)
                s = torch.cat(s, dim=0)
                i = torch.cat(i, dim=0)

                nms_idx = batched_nms(b, s, i, self.nmsThreshold)
                b = clip_boxes_to_image(b[nms_idx], size=(w, h)).int()
                i = i[nms_idx]

                return b, i
            else:
                return None

    def _second_stage(self,
                      imgs: torch.Tensor,
                      p_bbs: torch.Tensor,
                      p_idxs: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        _imgs = self._gather_rois(imgs, p_bbs, p_idxs, 24)

        with EvalScope(self.rNet):
            reg, pro = self.rNet(_imgs)

            mask = torch.ge(pro[:, 1], self.rNetThreshold)

            if not mask.any():
                return None

            reg = reg[mask]
            pro = pro[:, 1][mask]
            b = p_bbs[mask].type(torch.float32)
            i = p_idxs[mask]

            b = self._bb_reg(b, reg)
            j = batched_nms(b, pro, i, self.nmsThreshold)
            b = clip_boxes_to_image(b[j], size=imgs.shape[2:]).int()
            i = i[j]

            return b, i

    def _third_stage(self,
                     imgs: torch.Tensor,
                     r_bbs: torch.Tensor,
                     r_idxs: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        _imgs = self._gather_rois(imgs, r_bbs, r_idxs, 48)

        with EvalScope(self.oNet):
            reg, lmk, pro = self.oNet(_imgs)
            mask = torch.ge(pro[:, 1], self.oNetThreshold)

            if not mask.any():
                return None

            reg = reg[mask]
            pro = pro[:, 1][mask]
            b = r_bbs[mask].type(torch.float32)
            i = r_idxs[mask]

            b = self._bb_reg(b, reg)
            j = batched_nms(b, pro, i, self.nmsThreshold)
            b = clip_boxes_to_image(b[j], size=imgs.shape[2:]).int()
            i = i[j]

            return b, i, lmk[j]

    img_t = Union[np.ndarray, torch.Tensor]
    inp_t = Union[Sequence[img_t], img_t]

    def detect(self, imgs: inp_t):
        """ Perform detection on given image(s)

        Args:
            imgs: Single or multiple images. The image must be un-normalized,
             and in RGB order.

        Returns:
            A tuple of three tensors:
                (1) bounding boxes tensor
                (2) Image index tensor
                (3) Landmark tensors
        """
        _imgs = self.prepare_input(imgs=imgs, device=self.device)

        p_bbs, p_idxs = self._first_stage(_imgs)

        if len(p_idxs) <= 0:
            return None

        r_bbs, r_idxs = self._second_stage(_imgs, p_bbs, p_idxs)

        if len(r_idxs) <= 0:
            return None

        o_bbs, o_idxs, o_lmks = self._third_stage(_imgs, r_bbs, r_idxs)

        if len(o_idxs) <= 0:
            return None

        return o_bbs, o_idxs, o_lmks

    @staticmethod
    def prepare_single_img(img: img_t) -> torch.Tensor:
        """ Prepare a single image, used by `prepare_input` function.

        Args:
            img: Non-normalized RGB image, unsigned 8 bits integer

        Returns:
            Image converted to torch.Tensor type if necessary. The shape of the
            output tensor is `[1, h, w, c]`
        """
        if isinstance(img, np.ndarray):
            img = torch.tensor(img)

        if img.ndim == 3:
            img = img.unsqueeze(0)

        return img

    @staticmethod
    def prepare_input(imgs: inp_t,
                      device: torch.device) -> torch.Tensor:
        """ Prepare the input to be ready for detect method.

        Args:
            imgs: Non-normalized RGB image(s), unsigned 8 bits integer
            device: device where output tensor is stored

        Returns:
            Normalized, permuted, concatenated images with shape `[b, c, h, w]`
        """
        if isinstance(imgs, Sequence):
            if any(img.shape != imgs[0].shape for img in imgs[1:]):
                raise Exception("All images must be equal in dimensions.")
            imgs = [MTCNN.prepare_single_img(img) for img in imgs]
            imgs = torch.cat(imgs, dim=0)
        else:
            imgs = MTCNN.prepare_single_img(imgs)

        imgs = imgs.permute(0, 3, 1, 2).to(device)
        imgs = imgs / 255. - .5
        imgs = imgs.type(torch.float32)

        return imgs
