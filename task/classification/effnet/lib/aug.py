from torchvision import transforms
import torchvision.transforms.functional as F 
import random
import cv2
import numpy as np
from PIL import ImageEnhance

def random_pick(arr, probabilities):
    x = random.uniform(0,1) 
    cumulative_probability = 0.0 
    for item, item_probability in zip(arr, probabilities): 
        cumulative_probability += item_probability 
        if x < cumulative_probability:break 
    return item 

def maker_border_resize(img, resolution):
    height, width = img.shape[0], img.shape[1]
    top, bottom, left, right = 0, 0, 0, 0
    delta = (height - width)
    ratio = resolution / max(max(height, width), 1.0)
    if delta >= 0:
        if delta % 2 == 0:
            left = right = delta // 2
        else:
            left = delta // 2
            right = delta // 2 + 1
    else:
        if delta % 2 == 0:
            top = bottom = -delta // 2
        else:
            top = -delta // 2
            bottom = -delta // 2 +1
    # print("top, bottom, left, right,",top, bottom, left, right)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    # print("img shape",img.shape)
    img = cv2.resize(img, (resolution, resolution), interpolation = cv2.INTER_LINEAR)
    # cv2.imwrite("after_resize.png",img)
    return img

def random_4s_rotate(img1, img2):
    angle = np.random.choice([0, 90, 180, -90])
    img1 = img1.rotate(angle)
    img2 = img2.rotate(angle)
    return img1, img2

def random_crop(img1, img2):
    left = np.random.randint(32)
    up = np.random.randint(32)
    cropped_img1 = img1.crop([left, up, left + 224, up + 224])
    cropped_img2 = img2.crop([left, up, left + 224, up + 224])
    return cropped_img1, cropped_img2

def random_rotate(img1, img2=None):
    angle = transforms.RandomRotation.get_params([-180, 180])
    img1 = img1.rotate(angle)
    if img2 is not None:
        img2 = img2.rotate(angle)
        return img1, img2
    else:
        return img1

def random_flip(img1, img2=None):
    if random.random() > 0.5:
        img1 = F.hflip(img1)
        if img2 is not None:
            img2 = F.hflip(img2)
    if random.random() > 0.5:
        img1 = F.vflip(img1)
        if img2 is not None:
            img2 = F.vflip(img2)
    if img2 is not None:
        return img1, img2
    else:
        return img1

def randomColor(img1, img2=None):
    """
    对图像进行颜色抖动
    """
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(img1).enhance(random_factor)  # 调整图像的饱和度
    if img2 is not None:
        color_image2 = ImageEnhance.Color(img2).enhance(random_factor)
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    if img2 is not None:
        brightness_image2 = ImageEnhance.Brightness(color_image2).enhance(random_factor)
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    if img2 is not None:
        contrast_image2 = ImageEnhance.Contrast(brightness_image2).enhance(random_factor)
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    sharp_image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度
    if img2 is not None:
        sharp_image2 = ImageEnhance.Sharpness(contrast_image2).enhance(random_factor)
    if img2 is not None:
        return sharp_image, sharp_image2
    else:
        return sharp_image

if __name__ == "__main__":
    from PIL import Image
    img1 = Image.open('../ori_train_data/RS@JK351B@0104_14@14@A@20190104143139@P@10@M.jpg')
    img2 = Image.open('../ori_train_data/RS@JK351B@0104_237@237@A@20190104191825@P@10@M.jpg')

    img1 = maker_border_resize(np.asarray(img1), 256)
    img2 = maker_border_resize(np.asarray(img2), 256)

    img1 = Image.fromarray(img1)
    img2 = Image.fromarray(img2)

    for i in range(50):
        img_t1, img_t2 = random_4s_rotate(img1, img2)
        img_t1, img_t2 = random_crop(img_t1, img_t2)
        cv2.imshow('img1', np.asarray(img_t1))
        cv2.imshow('img2', np.asarray(img_t2))
        cv2.waitKey(0)