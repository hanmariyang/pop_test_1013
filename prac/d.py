import torch
import cv2
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
imgs = ['https://ultralytics.com/images/zidane.jpg']

results = model(imgs)

results.save()