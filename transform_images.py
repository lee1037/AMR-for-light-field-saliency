import numpy as np
import random
import numbers
import functional as F
from PIL import Image
import random
class RandomCrop(object):


    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, imglist):


        imgcroplist = []
        index = 0
        for img in imglist:
            if self.padding is not None:
                img = F.pad(img, self.padding, self.fill, self.padding_mode)

            # pad the width if needed
            if self.pad_if_needed and img.size[0] < self.size[1]:
                img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
            # pad the height if needed
            if self.pad_if_needed and img.size[1] < self.size[0]:
                img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
            if index == 0:
                i, j, h, w = self.get_params(img, self.size)
                index = 1

            imgcroplist.append(img.crop((j, i, j + w, i + h)))

        return imgcroplist

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imglist):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        imgfliplist = []
        if random.random() < self.p:
            for img in imglist:
                imgfliplist.append(img.transpose(Image.FLIP_LEFT_RIGHT))
            return imgfliplist
        return imglist

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imglist):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        imgfliplist = []
        if random.random() < self.p:
            for img in imglist:
                imgfliplist.append(img.transpose(Image.FLIP_TOP_BOTTOM))
            return imgfliplist
        return imglist

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class Normalize(object):
    """Vertically flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, imglist):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        imgnormlist = []
        for img in imglist:
            img = np.array(img, dtype=np.float32) / 255
            img -= self.mean
            img /= self.std
            img = img.transpose(2, 0, 1)
            imgnormlist.append(img)
        return imgnormlist

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class Resize(object):
    """Vertically flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, imglist):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        imgnormlist = []
        for img in imglist:
            imgnormlist.append(img.resize((self.size,self.size)))
        return imgnormlist

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomRotation(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, imglist):
        """
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """
        imgrotationlist = []
        angle = self.get_params(self.degrees)
        for img in imglist:
            imgrotationlist.append(F.rotate(img, angle, self.resample, self.expand, self.center))

        return imgrotationlist

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string

class Rotation(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
    """

    def __init__(self, resample=False, expand=False, center=None):

        self.resample = resample
        self.expand = expand
        self.center = center



    def __call__(self, imglist):
        """
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """
        imgrotationlist = []
        anglelist = [0, 90, 180, 270]
        index = random.randint(0, 3)
        angle = anglelist[index]
        for img in imglist:
            imgrotationlist.append(F.rotate(img, angle, self.resample, self.expand, self.center))

        return imgrotationlist

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string