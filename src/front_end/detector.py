import argparse
from sys import platform

from yolov3.models import *  # set ONNX_EXPORT in models.py
from yolov3.utils.datasets import *
from yolov3.utils.utils import *

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/coral/python')
from coral.tflite.python.examples.detection import detect as tpu_detect
from pycoral.utils import edgetpu
from pycoral.adapters import common
import numpy as np
from PIL import Image
from PIL import ImageDraw
import pdb


# based on https://github.com/google-coral/tflite/blob/master/python/examples/detection/detect.py
# max_instances_per_class can be a list with a number for each class, or a number applied to all classes
# Returns (x,y,w,h, object_conf, class_conf, class)
class EdgeTPU:
    def __init__(self, sample_im, model_dir='/mounted_folder/models', model_name='ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite', img_size=416, conf_thres=0.5, classes_ids=[80], max_instances_per_class=5):
        # ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite   |   ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.classes_ids = classes_ids
        # if isinstance(max_instances_per_class, int):
        #     self.max_instances_per_class = [max_instances_per_class]*len(classes_ids)
        # elif len(max_instances_per_class)== len(classes_ids):
        #     self.max_instances_per_class = max_instances_per_class
        # else:
        #     raise NameError('Inconsistent max instances per class and classes ids')
        self.classes_ids = classes_ids
        
        # Initialize the TF interpreter
        model_file_path_and_name = os.path.join(model_dir, model_name)
        self.interpreter = edgetpu.make_interpreter(model_file_path_and_name)
        self.interpreter.allocate_tensors()
        self.size = common.input_size(self.interpreter)

        

        # self.label_file = os.path.join(model_dir, 'coco_labels.txt')
        # image_file = os.path.join(script_dir, 'parrot.jpg')

    def detect(self, img0):
        # Image needs to be PIL?
        print("in EDGE detector function")
        image_PIL = Image.fromarray(img0) # PIL format
        # image_PIL = Image.open("/mounted_folder/images/img_484.png")
        # scale = tpu_detect.set_input(self.interpreter, image_PIL.size, lambda size: image_PIL.resize(size, Image.ANTIALIAS))
        def tmp_func(size):
            return image_PIL.resize(size, Image.ANTIALIAS)
        scale = tpu_detect.set_input(self.interpreter, image_PIL.size, tmp_func)
        objs = tpu_detect.get_output(self.interpreter, self.conf_thres, scale)

        det = [o for o in objs if o.id in self.classes_ids]

        if len(det) == 0:
            print('No objects detected')
            return np.array([])
        pdb.set_trace()
        det = np.stack(det)


        # Reformat det to x,y,w,h (x and y are top left corner's position)

        det[:,2] = det[:,2] - det[:,0]
        det[:,3] = det[:,3] - det[:,1]
        return det



# Adapted from detect.py of https://github.com/ultralytics/yolov3
# max_instances_per_class can be a list with a number for each class, or a number applied to all classes
# Returns (x,y,w,h, object_conf, class_conf, class)
class YoloDetector:
    def __init__(self,sample_im, base_dir='', cfg='yolov3/cfg/yolov3.cfg',weights='yolov3/weights/yolov3.weights',img_size=416,conf_thres=0.5,nms_thres=0.5,half=True,classes_ids=[80], max_instances_per_class = 5):
        # TODO Make sure this is valid
        self.img_size = 416#sample_im.shape[:2]
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.half = half
        self.classes_ids = classes_ids
        if isinstance(max_instances_per_class,int):
            self.max_instances_per_class = [max_instances_per_class]*len(classes_ids)
        elif len(max_instances_per_class)== len(classes_ids):
            self.max_instances_per_class = max_instances_per_class
        else:
            raise NameError('Inconsistent max instances per class and classes ids')

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
            
        with torch.no_grad():
            pred = self.model(img)[0]

        if self.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.nms_thres)
        if not torch.is_tensor(pred[0]):
            return np.array([])
            print('No object detected')

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        det = det.cpu().detach().numpy()


        # Keep only classes we want
        # det = det[np.isin(det[:,-1],self.classes_ids)]

        # Keep only certain number of instance per class
        if len(det) == 0:
            return np.array([])
            print('No object detected')
        else:
            det_filtered = []
            for i,class_id in enumerate(self.classes_ids):
                det_filtered.extend(det[det[:,-1] == class_id][-self.max_instances_per_class[i]:,:])

        det = det_filtered

        if len(det) == 0:
            return np.array([])
            print('No object of the chosen classes detected')

        det = np.stack(det)


        # Reformat det to x,y,w,h (x and y are top left corner's position)

        det[:,2] = det[:,2] - det[:,0]
        det[:,3] = det[:,3] - det[:,1]
        return det


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


# if __name__ == '__main__':
#     print("THIS IS FOR DEBUG ONLY")
#     try:
#         np.set_printoptions(linewidth=160, suppress=True)  # format numpy so printing matrices is more clear

#         d = EdgeTPU(model_dir='/mounted_folder/models/', 
#                          model='ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite', 
#                          classes_ids=[80], 
#                          max_instances_per_class=5)


#         # tmp_img = cv2.imread("/mounted_folder/images/img_556.png")
#         image_file = os.path.join("/mounted_folder/images", 'img_556.png')
#         # size = common.input_size(d.interpreter)
#         tmp_img = Image.open(image_file).convert('RGB').resize(d.size, Image.ANTIALIAS)
#         det = d.detect(tmp_img)

#         pdb.set_trace()
#         # Print the result
#         labels = dataset.read_label_file(d.label_file)  # a dictionary the converts class id to string (e.g. '2' to 'car')
#         for c in det:
#             print('%s: %.5f' % (labels.get(c.id, c.id), c.score))
#     except:
#         import traceback
#         traceback.print_exc()
#     print("done with detect test!")