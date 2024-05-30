import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch
import cv2
import numpy as np
import requests
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from task.object_detection.yolox import  yolox_predictor
from task.object_detection.yolox.lib.utils.visualize import vis

COLORS = np.random.uniform(0, 255, size=(80, 3))

def parse_detections(results):
    detections = results.pandas().xyxy[0]
    detections = detections.to_dict()
    boxes, colors, names = [], [], []

    for i in range(len(detections["xmin"])):
        confidence = detections["confidence"][i]
        if confidence < 0.2:
            continue
        xmin = int(detections["xmin"][i])
        ymin = int(detections["ymin"][i])
        xmax = int(detections["xmax"][i])
        ymax = int(detections["ymax"][i])
        name = detections["name"][i]
        category = int(detections["class"][i])
        color = COLORS[category]

        boxes.append((xmin, ymin, xmax, ymax))
        colors.append(color)
        names.append(name)
    return boxes, colors, names


def draw_detections(boxes, colors, names, img):
    for box, color, name in zip(boxes, colors, names):
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(
            img,
            (xmin, ymin),
            (xmax, ymax),
            color,
            2)

        cv2.putText(img, name, (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return img


# image_path = "./puppies.jpg"
# img = cv2.imread(image_path)
# # img = np.array(Image.open(image_path))
# img = cv2.resize(img, (640, 640))
# rgb_img = img.copy()
# img = np.float32(img) / 255
# transform = transforms.ToTensor()
# tensor = transform(img).unsqueeze(0)
#
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# model.eval()
# model.cpu()
# target_layers = [model.model.model.model[-2]]
#
# results = model([rgb_img])
# boxes, colors, names = parse_detections(results)
# detections = draw_detections(boxes, colors, names, rgb_img.copy())

# cv2.imshow("detection", detections)
# cv2.waitKey()


# cam = EigenCAM(model, target_layers, use_cuda=False)
# grayscale_cam = cam(tensor)[0, :, :]
# cam_image = show_cam_on_image(np.float32(detections) / 255, grayscale_cam, use_rgb=False)
# cv2.imshow("cam", cam_image)
# cv2.waitKey()

model_dir = "./outputs/MS_HR"
preditor = yolox_predictor.YoloxPredictor()
preditor.loadModel(model_dir)
model = preditor.model
target_layers = [model.head.reg_convs[0]]

img_pth = "./datasets/cam/ori.jpg"
img = cv2.imread(img_pth)
bgr_img = img.copy()
img = cv2.resize(img, (640, 640))
img = np.float32(img) / 255
transform = transforms.ToTensor()
img_tensor = transform(img).unsqueeze(0)

results = preditor.predict(bgr_img)
detection = vis(bgr_img, results["results"], class_names=preditor.cls_names)
# detection = cv2.resize(detection, (1024, 1024))
# cv2.imshow("detection", detection)
# cv2.waitKey()
# cv2.imwrite("./datasets/3_outputs.jpg", detection)

detection = cv2.resize(detection, (640, 640))

cam = EigenCAM(model, target_layers, use_cuda=True)
grayscale_cam = cam(img_tensor)[0, :, :]
cam_image = show_cam_on_image(np.float32(detection) / 255, grayscale_cam, use_rgb=False)
# cv2.imshow("cam", cam_image)
# cv2.waitKey()
cv2.imwrite("./datasets/cam/cam0.jpg", cam_image)