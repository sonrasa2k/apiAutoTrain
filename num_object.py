import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import random

class NumObject:
    def __init__(self):
        path_img = True
    def num_object(self,path_img,path_modeled,name_object,save_path):
        # Load model
        weights = path_modeled
        set_logging()
        device = select_device('')
        half = device.type != 'cpu'
        imgsz = 640


        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        source = path_img
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        if half:
            model.half()  # to FP16
        # Get names and colors
        names = name_object
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        conf_thres = 0.3
        iou_thres = 0.25

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)

            extra = ""
            # Process detections

            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                save_path = save_path + "detect.jpg"
                if len(det):
                    print(len(det))
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    print(im0.shape[0])
                    width_img = im0.shape[0]
                    height_img = im0.shape[1]
                    # Write results
                    boxes = []
                    for *xyxy, conf, cls in reversed(det):
                        boxx = [float(xyxy[0].item()/width_img),float(xyxy[1].item()/height_img),float(xyxy[2].item()/width_img),float(xyxy[3].item()/height_img),conf.item()]
                        boxes.append(boxx)
                        if save_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                            # extra += "<br>- <b>" + str(names[int(cls)]) + "</b> (" + dict[names[int(cls)]] + ") với độ tin cậy <b>{:.2f}% </b>".format(conf)

                # Save results (image with detections)
                if save_img:
                    cv2.imwrite(save_path, im0)
                    print("da luu file : {0}".format(save_path))
        if len(det):
            return {"name_object":name_object,"num":len(det),"data":boxes}
        return {"name_object":name_object,"num":0,"data":[]}



if __name__ == '__main__':
    a = NumObject()
    b = a.num_object(r'fanta3.jpg','run/train/son/last.pt','fanta','runs/detect/')
    print(b)