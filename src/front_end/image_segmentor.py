from detector import YoloDetector
from tracker import SiammaskTracker
import sys, os, time

class ImageSegmentor:
    def __init__(self,sample_im,detector_name='yolov3',tracker_name='siammask', detect_class_ids=[80],use_trt=False):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/front_end/'
        if detector_name == 'yolov3':
            self.detector = YoloDetector(sample_im,base_dir=base_dir, classes_ids=detect_class_ids)
        else:
            raise RuntimeError("Detector chosen not implemented")

        if tracker_name == 'siammask':
            self.tracker = SiammaskTracker(sample_im,base_dir=base_dir, use_tensorrt=use_trt)
        else:
            raise RuntimeError("Tracker chosen not implemented")

        self.last_boxes = []
        self.last_classes = []
        
    def track(self,image):
        tic = time.time()
        self.last_boxes,_, self.last_classes = self.tracker.track(image)
        tic2 = time.time()
        print("track time = {:.4f}".format(tic2- tic))
        return self.last_boxes, self.last_classes

    def reinit_tracker(self,new_boxes,image):
        self.tracker.reinit(new_boxes,image)

    def detect(self,image):
        tic = time.time()
        detections = self.detector.detect(image)
        print("detect time = {:.4f}".format(time.time() - tic))
        return detections