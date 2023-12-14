import torch
import torch.nn as nn
from DiffJPEG.compression import compress_jpeg
from DiffJPEG.decompression import decompress_jpeg
from DiffJPEG.utils import quality_to_factor
import random
from torchvision import transforms
from torchvision.transforms import GaussianBlur

class AddNoise(nn.Module):
    def __init__(self, noise_type="gaussian"):
        super(AddNoise, self).__init__()
        self.noise_type = noise_type

    def forward(self, x):
        if self.noise_type == "gaussian":
            noise = torch.randn(x.size()) * 0.1
        elif self.noise_type == "salt_pepper":
            noise = torch.rand(x.size())
            noise[noise < 0.1] = 0  # pepper
            noise[noise > 0.9] = 1  # salt

        noise = noise.to(x.device)
        return x + noise

class BlurImage(nn.Module):
    def __init__(self):
        super(BlurImage, self).__init__()
        self.blur = GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))

    def forward(self, x):
        return self.blur(x)

class ColorJitterTransform(nn.Module):
    def __init__(self):
        super(ColorJitterTransform, self).__init__()
        self.transform = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)

    def forward(self, x):
        return self.transform(x)

class DiffJPEG(nn.Module):
    '''
       reference: https://github.com/mlomnitz/DiffJPEG
    '''

    def __init__(self, quality=80):
        ''' Initialize the DiffJPEG layer
        Inputs:
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme.
        '''
        super(DiffJPEG, self).__init__()
        factor = quality_to_factor(quality)
        from DiffJPEG.utils import diff_round
        rounding = diff_round
        self.compress = compress_jpeg(rounding=rounding, factor=factor)
        self.decompress = decompress_jpeg(factor=factor)

    def forward(self, x):
        '''
        '''
        b, c, h, w = x.shape
        y, cb, cr = self.compress(x)
        recovered = self.decompress(y, h, w, cb, cr)
        return recovered


class DiffNone(nn.Module):
    def __init__(self):
        super(DiffNone, self).__init__()

    def forward(self, x):
        return x


class DiffBitReduction(nn.Module):
    def __init__(self, step_num=8):
        super(DiffBitReduction, self).__init__()
        self.step_num = step_num

    def forward(self, x):
        quantized_x = torch.round(x / self.step_num) * self.step_num
        quantized_x = torch.clamp(quantized_x, min=0, max=255)
        return quantized_x + x - x.detach()


class DiffRandomization(nn.Module):
    def __init__(self, scale_factor=0.9):
        super(DiffRandomization, self).__init__()
        self.scale_factor = scale_factor
        random.seed(1234)

    def forward(self, x):
        ori_size = x.size()[-2:]
        scale_factor = random.uniform(self.scale_factor, 1)
        x = nn.functional.interpolate(x, scale_factor=scale_factor)
        new_size = x.size()[-2:]

        delta_w = ori_size[1] - new_size[1]
        delta_h = ori_size[0] - new_size[0]
        top = random.randint(0, delta_h + 1)
        left = random.randint(0, delta_w + 1)
        bottom, right = delta_h - top, delta_w - left

        x = nn.functional.pad(x, pad=(left, right, top, bottom), value=0)
        return x


transform_modules = {
    'JPEG': DiffJPEG,
    'None': DiffNone,
    'BitReudction': DiffBitReduction,
    'Randomization': DiffRandomization,
}