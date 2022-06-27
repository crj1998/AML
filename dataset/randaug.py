

import random
from PIL import Image, ImageEnhance, ImageOps, ImageFilter


MAX = 10


def float_v(v, vmin, vmax):
    return float(vmin + (vmax-vmin) / MAX * v)

def int_v(v, vmin, vmax):
    return round(vmin + (vmax-vmin) / MAX * v)



########## Color change ##########

def AutoContrast(img, **kwarg):
    return ImageOps.autocontrast(img)

def Brightness(img, v, vmin, vmax, **kwarg):
    """ Adjust image brightness.
    """
    v = float_v(v, vmin, vmax)
    return ImageEnhance.Brightness(img).enhance(v)

def Color(img, v, vmin, vmax, **kwarg):
    """ Adjust image color balance.
    """
    v = float_v(v, vmin, vmax)
    return ImageEnhance.Color(img).enhance(v)

def Contrast(img, v, vmin, vmax, **kwarg):
    """ Adjust image contrast.
    """
    v = float_v(v, vmin, vmax)
    return ImageEnhance.Contrast(img).enhance(v)

def Sharpness(img, v, vmin, vmax, **kwarg):
    """ Adjust image sharpness.
    """
    v = float_v(v, vmin, vmax)
    return ImageEnhance.Sharpness(img).enhance(v)

def Equalize(img, **kwarg):
    """ Equalize the image histogram
    """ 
    return ImageOps.equalize(img)

def Solarize(img, v, vmin, vmax, **kwarg):
    """ Invert all pixel values above a threshold.
    """
    v = int_v(v, vmin, vmax)
    return ImageOps.solarize(img, v)

def Posterize(img, v, vmin, vmax, **kwarg):
    """ Reduce the number of bits for each color channel.
    """
    v = int_v(v, vmin, vmax)
    return ImageOps.posterize(img, v)

def Channelshuffle(img, **kwargs):
    chs = list(img.split())
    random.shuffle(chs)
    return Image.merge('RGB', chs)


######## Shape Transform #######

def HorizontalFlip(im, **kwargs):
    return ImageOps.mirror(im)

def Crop(im, v, vmin, vmax, **kwargs):
    W, H = im.size
    w, h = round(W*random.uniform(vmin, vmax)), round(H*random.uniform(vmin, vmax))
    x, y = random.randint(0, W - w), random.randint(0, H - h)
    return im.crop((x, y, x+w, y+h)).resize((W, H))

def Rotate(img, v, vmin, vmax, **kwarg):
    v = int_v(v, vmin, vmax)
    return img.rotate(v)

def ShearX(img, v, vmin, vmax, **kwarg):
    v = float_v(v, vmin, vmax)
    return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(img, v, vmin, vmax, **kwarg):
    v = float_v(v, vmin, vmax)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))

def TranslateX(img, v, vmin, vmax, **kwarg):
    v = float_v(v, vmin, vmax)
    v = round(v * img.size[0])
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v, vmin, vmax, **kwarg):
    v = float_v(v, vmin, vmax)
    v = round(v * img.size[1])
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


######### Filter operation #########

def MedianFilter(img, v, vmin, vmax, **kwarg):
    v = int_v(v, vmin, vmax)//2 * 2 + 1
    return img.filter(ImageFilter.MedianFilter(v))

def MinFilter(img, v, vmin, vmax, **kwarg):
    v = int_v(v, vmin, vmax)//2 * 2 + 1
    return img.filter(ImageFilter.MinFilter(v))

def MaxFilter(img, v, vmin, vmax, **kwarg):
    v = int_v(v, vmin, vmax)//2 * 2 + 1
    return img.filter(ImageFilter.MaxFilter(v))

def ModeFilter(img, v, vmin, vmax, **kwarg):
    v = int_v(v, vmin, vmax)//2 * 2 + 1
    return img.filter(ImageFilter.ModeFilter(v))



class RandAugment(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = [
            (AutoContrast, None, None), 
            (Brightness, 0.4, 2.0),
            (Color, 0.4, 2.0),
            (Contrast, 0.4, 2.0),
            (Sharpness, 0.4, 2.0),
            (Posterize, 2, 6), 
            (Equalize, None, None),
            [(MedianFilter, 1, 5), (MedianFilter, 1, 5), (MedianFilter, 1, 5), (ModeFilter, 1, 5)], 
            (Crop, 0.7, 1.0),
            [(Rotate, -25, 25), (ShearX, -0.25, 0.25), (ShearY, -0.25, 0.25), (TranslateX, -0.25, 0.25), (TranslateY, -0.25, 0.25)]
        ]


    def __repr__(self):
        augs = ", ".join([", ".join(j[0].__name__ for j in i) if isinstance(i, list) else i[0].__name__ for i in self.augment_pool])
        return f"RandAugmentMC(n={self.n}, m={self.m}) [{augs}]"

    def __str__(self):
        return self.__repr__()
    
    def __call__(self, img):
        for i in random.sample(self.augment_pool, k=random.randint(1, self.n+1)):
            op, vmin, vmax = random.choice(i) if isinstance(i, list) else i
            v = random.randint(0, self.m+1)
            img = op(img, v=v, vmin=vmin, vmax=vmax)
        return img