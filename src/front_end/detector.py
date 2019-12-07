import argparse
from sys import platform

from yolov3.models import *  # set ONNX_EXPORT in models.py
from yolov3.utils.datasets import *
from yolov3.utils.utils import *

# Adapted from detect.py of https://github.com/ultralytics/yolov3
class YoloDetector:
    def __init__(self, base_dir='', cfg='yolov3/cfg/yolov3.cfg',data='yolov3/data/coco.data',weights='yolov3/weights/yolov3.weights',img_size=416,conf_thres=0.3,nms_thres=0.5,half=True,class_id=0):
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.half = half

        # Initialize
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = Darknet(base_dir + cfg, img_size)

        # Load weights
        attempt_download(base_dir + weights)
        if weights.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(base_dir + weights, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(self.model, base_dir + weights)

        # Fuse Conv2d + BatchNorm2d layers
        # model.fuse()

        # Eval mode
        self.model.to(self.device).eval()

        # Half precision
        self.half = self.half and self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.model.half()

    def detect(self,img0):
        img = letterbox(img0, new_shape=self.img_size)[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        # Run inference
        # Get detections
        img = torch.from_numpy(img).to(self.device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img)[0]

        if self.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.nms_thres)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        # Check if wanted class found, and choose the one with highest confidence above threshold
        det = det[d[:,-1] == class_id]
        if len(det) == 0:
            return False
        else:
            det = det[torch.argmax(det[:,4])]

        # Reformat det to x,y,w,h (x and y are top left corner's position)
        det = det.cpu().detach().numpy()
        return [det[0],det[1],det[2]-det[0],det[3]-det[1]]




def letterbox(img, new_shape=(416, 416), color=(128, 128, 128),
            auto=True, scaleFill=False, scaleup=True, interp=cv2.INTER_AREA):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = max(new_shape) / max(shape)
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=interp)  # INTER_AREA is better, INTER_LINEAR is faster
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)