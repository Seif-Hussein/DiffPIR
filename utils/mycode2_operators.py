import math
from copy import deepcopy
from math import pi
from typing import Any, Dict, Tuple

import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import beta, triangular, uniform
from PIL import Image, ImageDraw, ImageFilter

from utils.utils_resizer import Resizer


MOTION_EPS = 0.1


def polar2z(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return r * np.exp(1j * theta)


class Kernel:
    """Motion blur kernel generator used by mycode2."""

    def __init__(self, size: tuple = (100, 100), intensity: float = 0):
        if not isinstance(size, tuple):
            raise ValueError("Size must be tuple of 2 positive integers")
        if len(size) != 2 or not isinstance(size[0], int) or not isinstance(size[1], int):
            raise ValueError("Size must be tuple of 2 positive integers")
        if size[0] < 0 or size[1] < 0:
            raise ValueError("Size must be tuple of 2 positive integers")
        if type(intensity) not in [int, float, np.float32, np.float64]:
            raise ValueError("Intensity must be a number between 0 and 1")
        if intensity < 0 or intensity > 1:
            raise ValueError("Intensity must be a number between 0 and 1")

        self.SIZE = size
        self.INTENSITY = intensity
        self.SIZEx2 = tuple(2 * i for i in size)
        self.x, self.y = self.SIZEx2
        self.DIAGONAL = (self.x**2 + self.y**2) ** 0.5
        self.kernel_is_generated = False

    def _createPath(self):
        self.MAX_PATH_LEN = 0.75 * self.DIAGONAL * (
            uniform() + uniform(0, self.INTENSITY**2)
        )
        steps = []
        while sum(steps) < self.MAX_PATH_LEN:
            step = beta(1, 30) * (1 - self.INTENSITY + MOTION_EPS) * self.DIAGONAL
            if step < self.MAX_PATH_LEN:
                steps.append(step)
        self.NUM_STEPS = len(steps)
        self.STEPS = np.asarray(steps)

        self.MAX_ANGLE = uniform(0, self.INTENSITY * pi)
        self.JITTER = beta(2, 20)
        angles = [uniform(low=-self.MAX_ANGLE, high=self.MAX_ANGLE)]
        while len(angles) < self.NUM_STEPS:
            angle = triangular(
                0,
                self.INTENSITY * self.MAX_ANGLE,
                self.MAX_ANGLE + MOTION_EPS,
            )
            if uniform() < self.JITTER:
                angle *= -np.sign(angles[-1])
            else:
                angle *= np.sign(angles[-1])
            angles.append(angle)
        self.ANGLES = np.asarray(angles)

        complex_increments = polar2z(self.STEPS, self.ANGLES)
        self.path_complex = np.cumsum(complex_increments)
        self.com_complex = sum(self.path_complex) / self.NUM_STEPS
        center_of_kernel = (self.x + 1j * self.y) / 2
        self.path_complex -= self.com_complex
        self.path_complex *= np.exp(1j * uniform(0, pi))
        self.path_complex += center_of_kernel
        self.path = [(i.real, i.imag) for i in self.path_complex]

    def _createKernel(self):
        if self.kernel_is_generated:
            return None
        self._createPath()
        self.kernel_image = Image.new("RGB", self.SIZEx2)
        painter = ImageDraw.Draw(self.kernel_image)
        painter.line(xy=self.path, width=int(self.DIAGONAL / 150))
        self.kernel_image = self.kernel_image.filter(
            ImageFilter.GaussianBlur(radius=int(self.DIAGONAL * 0.01))
        )
        self.kernel_image = self.kernel_image.resize(self.SIZE, resample=Image.LANCZOS)
        self.kernel_image = self.kernel_image.convert("L")
        self.kernel_is_generated = True

    @property
    def kernelMatrix(self) -> np.ndarray:
        self._createKernel()
        kernel = np.asarray(self.kernel_image, dtype=np.float32)
        kernel /= np.sum(kernel)
        return kernel


class Operator:
    def __init__(self, sigma: float):
        self.sigma = float(sigma)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def measure(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self(x)
        return y0 + self.sigma * torch.randn_like(y0)

    def initial_guess(self, y: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        if tuple(y.shape) == tuple(x_shape):
            return y
        return torch.randn(x_shape, device=y.device, dtype=y.dtype)

    def low_quality_image(self, y: torch.Tensor) -> torch.Tensor:
        return y.clamp(-1.0, 1.0)


class DownSampling(Operator):
    def __init__(
        self,
        resolution: int = 256,
        scale_factor: int = 4,
        device: str = "cuda",
        sigma: float = 0.05,
        channels: int = 3,
    ):
        super().__init__(sigma)
        self.resolution = int(resolution)
        self.scale_factor = int(scale_factor)
        self.channels = int(channels)
        in_shape = [1, self.channels, self.resolution, self.resolution]
        self.down_sample = Resizer(in_shape, 1 / self.scale_factor).to(device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_sample(x)

    def initial_guess(self, y: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        return F.interpolate(
            y,
            size=x_shape[-2:],
            mode="bicubic",
            align_corners=False,
        ).clamp(-1.0, 1.0)


def random_sq_bbox(img, mask_shape, image_size=256, margin=(32, 32)):
    b, c, h, w = img.shape
    mask_h, mask_w = mask_shape
    margin_height, margin_width = margin
    maxt = image_size - margin_height - mask_h
    maxl = image_size - margin_width - mask_w

    top = np.random.randint(margin_height, maxt)
    left = np.random.randint(margin_width, maxl)
    mask = torch.ones([b, c, h, w], device=img.device)
    mask[..., top : top + mask_h, left : left + mask_w] = 0
    return mask, top, top + mask_h, left, left + mask_w


class MaskGenerator:
    def __init__(
        self,
        mask_type,
        mask_len_range=None,
        mask_prob_range=None,
        image_size=256,
        margin=(32, 32),
    ):
        assert mask_type in ["box", "random", "both", "extreme", "whole"]
        self.mask_type = mask_type
        self.mask_len_range = mask_len_range
        self.mask_prob_range = mask_prob_range
        self.image_size = int(image_size)
        self.margin = margin

    def _retrieve_box(self, img):
        low, high = self.mask_len_range
        mask_h = np.random.randint(int(low), int(high))
        mask_w = np.random.randint(int(low), int(high))
        mask, top, bottom, left, right = random_sq_bbox(
            img,
            mask_shape=(mask_h, mask_w),
            image_size=self.image_size,
            margin=self.margin,
        )
        return mask, top, bottom, left, right

    def _retrieve_random(self, img):
        total = self.image_size**2
        low, high = self.mask_prob_range
        prob = np.random.uniform(low, high)
        mask_vec = torch.ones([1, self.image_size * self.image_size])
        samples = np.random.choice(
            self.image_size * self.image_size, int(total * prob), replace=False
        )
        mask_vec[:, samples] = 0
        mask_b = mask_vec.view(1, self.image_size, self.image_size)
        mask_b = mask_b.repeat(img.shape[1], 1, 1)
        mask = torch.ones_like(img, device=img.device)
        mask[:, ...] = mask_b.to(device=img.device, dtype=img.dtype)
        return mask

    def __call__(self, img):
        if self.mask_type == "random":
            return self._retrieve_random(img)
        if self.mask_type == "box":
            mask, _, _, _, _ = self._retrieve_box(img)
            return mask
        if self.mask_type == "extreme":
            mask, _, _, _, _ = self._retrieve_box(img)
            return 1.0 - mask
        if self.mask_type == "whole":
            return torch.zeros_like(img)
        raise NotImplementedError("mask_type='both' is registered for parity but not used here")


class Inpainting(Operator):
    supports_analytic_prox = True

    def __init__(
        self,
        mask_type,
        mask_len_range=None,
        mask_prob_range=None,
        resolution: int = 256,
        device: str = "cuda",
        sigma: float = 0.05,
    ):
        super().__init__(sigma)
        self.mask_gen = MaskGenerator(
            mask_type,
            mask_len_range=mask_len_range,
            mask_prob_range=mask_prob_range,
            image_size=resolution,
        )
        self.mask = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask is None:
            self.mask = self.mask_gen(x)
            self.mask = self.mask[0:1, 0:1, :, :]
        return x * self.mask

    def analytic_prox(self, x0: torch.Tensor, y: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return (self.mask * y + tau * x0).div(self.mask + tau)


class Blurkernel(nn.Module):
    def __init__(
        self,
        blur_type="gaussian",
        kernel_size=31,
        std=3.0,
        device=None,
        initialize=True,
    ):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = int(kernel_size)
        self.std = float(std)
        self.device = device
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size // 2),
            nn.Conv2d(3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3),
        )
        self.k = None
        if initialize:
            self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            impulse = np.zeros((self.kernel_size, self.kernel_size))
            impulse[self.kernel_size // 2, self.kernel_size // 2] = 1
            kernel = torch.from_numpy(scipy.ndimage.gaussian_filter(impulse, sigma=self.std))
        elif self.blur_type == "motion":
            kernel = torch.from_numpy(
                Kernel(size=(self.kernel_size, self.kernel_size), intensity=self.std).kernelMatrix
            )
        else:
            raise ValueError(f"Unknown blur type: {self.blur_type}")
        self.k = kernel
        for _, parameter in self.named_parameters():
            parameter.data.copy_(kernel)

    def update_weights(self, kernel):
        if not torch.is_tensor(kernel):
            kernel = torch.from_numpy(kernel).to(self.device)
        for _, parameter in self.named_parameters():
            parameter.data.copy_(kernel)

    def get_kernel(self):
        return self.k


class GaussianBlur(Operator):
    def __init__(self, kernel_size, intensity, device="cuda", sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.kernel_size = int(kernel_size)
        self.conv = Blurkernel(
            blur_type="gaussian",
            kernel_size=kernel_size,
            std=intensity,
            device=device,
        ).to(device)
        self.kernel = self.conv.get_kernel().to(dtype=torch.float32)
        self.conv.update_weights(self.kernel)
        self.conv.requires_grad_(False)

    def __call__(self, data):
        return self.conv(data)


class MotionBlur(Operator):
    def __init__(self, kernel_size, intensity, device="cuda", sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.kernel_size = int(kernel_size)
        self.conv = Blurkernel(
            blur_type="motion",
            kernel_size=kernel_size,
            std=intensity,
            device=device,
            initialize=False,
        ).to(device)
        self.kernel_object = Kernel(size=(self.kernel_size, self.kernel_size), intensity=intensity)
        self.kernel = torch.tensor(self.kernel_object.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(self.kernel)
        self.conv.requires_grad_(False)

    def __call__(self, data):
        return self.conv(data)


def fft2c_new(data: torch.Tensor) -> torch.Tensor:
    x = torch.view_as_complex(data)
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return torch.view_as_real(x)


def ifft2c_new(data: torch.Tensor) -> torch.Tensor:
    x = torch.view_as_complex(data)
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.ifft2(x, dim=(-2, -1), norm="ortho")
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return torch.view_as_real(x)


class PhaseRetrieval(Operator):
    def __init__(self, oversample=0.0, resolution=256, sigma=0.05):
        super().__init__(sigma)
        self.pad = int((float(oversample) / 8.0) * int(resolution))

    def __call__(self, x):
        x01 = x * 0.5 + 0.5
        x01 = F.pad(x01, (self.pad, self.pad, self.pad, self.pad))
        if not torch.is_complex(x01):
            x01 = x01.type(torch.complex64)
        fft2_m = torch.view_as_complex(fft2c_new(torch.view_as_real(x01)))
        return fft2_m.abs()

    def forward_complex(self, x01: torch.Tensor) -> torch.Tensor:
        x = F.pad(x01, (self.pad, self.pad, self.pad, self.pad))
        x_c = x.to(torch.complex64) if not torch.is_complex(x) else x
        return torch.view_as_complex(fft2c_new(torch.view_as_real(x_c)))

    def adjoint_complex(self, p: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        p_c = p.to(torch.complex64) if not torch.is_complex(p) else p
        x_pad_c = torch.view_as_complex(ifft2c_new(torch.view_as_real(p_c)))
        h, w = out_hw
        return x_pad_c[..., self.pad : self.pad + h, self.pad : self.pad + w].real

    @torch.no_grad()
    def proj_amplitude(
        self,
        x: torch.Tensor,
        y_amp: torch.Tensor,
        *,
        tau: float = float("inf"),
        eps_value: float = 1e-8,
        enforce_real: bool = True,
        clamp_x: bool = True,
        clamp01: bool = True,
    ) -> torch.Tensor:
        x01 = x * 0.5 + 0.5
        if clamp01:
            x01 = x01.clamp(0.0, 1.0)
        if self.pad > 0:
            x01 = F.pad(x01, (self.pad, self.pad, self.pad, self.pad))

        x01_c = x01.to(torch.complex64) if not torch.is_complex(x01) else x01
        u = torch.view_as_complex(fft2c_new(torch.view_as_real(x01_c)))
        mag = u.abs()
        y = y_amp.to(device=mag.device, dtype=mag.dtype)
        mag_new = y if math.isinf(tau) else (mag + tau * y) / (1.0 + tau)
        u_new = u * (mag_new / (mag + eps_value))

        x01_new_c = torch.view_as_complex(ifft2c_new(torch.view_as_real(u_new)))
        x01_new = x01_new_c.real if enforce_real else x01_new_c
        if self.pad > 0:
            x01_new = x01_new[..., self.pad : -self.pad, self.pad : -self.pad]
        if clamp01 and not torch.is_complex(x01_new):
            x01_new = x01_new.clamp(0.0, 1.0)
        x_new = x01_new * 2.0 - 1.0
        if clamp_x and not torch.is_complex(x_new):
            x_new = x_new.clamp(-1.0, 1.0)
        return x_new

    def low_quality_image(self, y: torch.Tensor) -> torch.Tensor:
        y_view = y.detach()
        flat = y_view.flatten(1)
        lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
        hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
        return ((y_view - lo) / (hi - lo + 1e-8) * 2.0 - 1.0).clamp(-1.0, 1.0)

    def get_more_aligned_option(self, template, options):
        template = deepcopy(template)
        options = deepcopy(options)
        template = ((template.detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
        result_lst = []
        for option in options:
            option = ((option.detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
            result_lst.append(((template - option) ** 2).sum())
        return result_lst.index(min(result_lst))


OPERATORS = {
    "down_sampling": DownSampling,
    "inpainting": Inpainting,
    "gaussian_blur": GaussianBlur,
    "motion_blur": MotionBlur,
    "phase_retrieval": PhaseRetrieval,
}


def get_operator(name: str, **kwargs):
    try:
        operator_cls = OPERATORS[name]
    except KeyError as exc:
        raise NameError(f"Operator {name} is not registered") from exc
    return operator_cls(**kwargs)


def build_operator(operator_config: Dict[str, Any], device: torch.device):
    kwargs = dict(operator_config)
    name = kwargs.pop("name")
    if name != "phase_retrieval":
        kwargs.setdefault("device", str(device))
    return get_operator(name=name, **kwargs)
