import random
import cv2
import numpy as np
import math


class VerticalFlip:
    def __init__(self, prob=.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 0)
            if mask is not None:
                mask = cv2.flip(mask, 0)
        return img, mask


class HorizontalFlip:
    def __init__(self, prob=.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)
        return img, mask


class RandomFlip:
    def __init__(self, prob=.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d)
            if mask is not None:
                mask = cv2.flip(mask, d)
        return img, mask


class RandomRotate90:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            factor = random.randint(0, 4)
            img = np.rot90(img, factor)
            if mask is not None:
                mask = np.rot90(mask, factor)
        return img.copy(), mask.copy()


class Rotate:
    def __init__(self, limit=20, prob=.5):
        self.prob = prob
        self.limit = limit

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            angle = random.uniform(-self.limit, self.limit)
            height, width = img.shape[0:2]
            mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
            img = cv2.warpAffine(img, mat, (width, height),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv2.warpAffine(mask, mat, (width, height),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)
        return img.copy(), mask.copy()


class Shift:
    def __init__(self, limit=4, prob=.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            limit = self.limit
            dx = round(random.uniform(-limit, limit))
            dy = round(random.uniform(-limit, limit))

            if np.ndim(img) > 2:
                height, width, channel = img.shape
            else:
                height, width = img.shape
            y1 = limit+1+dy
            y2 = y1 + height
            x1 = limit+1+dx
            x2 = x1 + width

            img1 = cv2.copyMakeBorder(img, limit+1, limit+1, limit+1, limit+1,
                                      borderType=cv2.BORDER_REFLECT_101)
            img = img1[y1:y2, x1:x2]
            if mask is not None:
                msk1 = cv2.copyMakeBorder(mask, limit+1, limit+1, limit+1,
                                          limit+1,
                                          borderType=cv2.BORDER_REFLECT_101)
                mask = msk1[y1:y2, x1:x2]
        return img, mask


class ShiftScaleRotate:
    def __init__(self, shift_limit=0.0625, scale_limit=0.0, rotate_limit=45,
                 prob=0.5):
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            height, width, channel = img.shape

            angle = random.uniform(-self.rotate_limit, self.rotate_limit)
            scale = random.uniform(1-self.scale_limit, 1+self.scale_limit)
            dx = round(random.uniform(-self.shift_limit, self.shift_limit)) \
                * width
            dy = round(random.uniform(-self.shift_limit, self.shift_limit)) \
                * height

            cc = math.cos(angle/180*math.pi) * scale
            ss = math.sin(angle/180*math.pi) * scale
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0],  [width, height], [0, height],
                             ])
            box1 = box0 - np.array([width/2, height/2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width/2+dx,
                                                            height/2+dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)
            img = cv2.warpPerspective(img, mat, (width, height),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv2.warpPerspective(mask, mat, (width, height),
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_REFLECT_101)
        return img, mask


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for ii, tform in enumerate(self.transforms):
            image, mask = tform()(image, mask)
        return image, mask


transforms = [VerticalFlip, HorizontalFlip, RandomFlip, RandomRotate90, Rotate,
              Shift]


class PairedTransforms:
    def __init__(self, num_transforms=0.5, transforms=transforms):
        self.transforms = random.choices(transforms,
                                         k=int(num_transforms*len(transforms)))

    def __call__(self, image, mask):
        return Compose(self.transforms)(image, mask)
