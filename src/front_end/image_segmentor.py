from detector import YoloDetector
from tracker import SiammaskTracker
import sys, os, time

class ImageSegmentor:
    def __init__(self,sample_im,detector_name='yolov3',tracker_name='siammask'):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/front_end/'
        if detector_name == 'yolov3':
            self.detector = YoloDetector(sample_im,base_dir=base_dir)
        else:
            raise RuntimeError("Detector chosen not implemented")

        if tracker_name == 'siammask':
            self.tracker = SiammaskTracker(sample_im,base_dir=base_dir)
        else:
            raise RuntimeError("Tracker chosen not implemented")

        self.last_box = None
        
    def track(self,image):
        tic = time.clock()
        self.last_box,_ = self.tracker.track(image)
        print("track time = {:.4f}".format(time.clock() - tic))
        return self.last_box

    def reinit_tracker(self,new_box,image):
        self.tracker.reinit(new_box,image)

    def detect(self,image):
        tic = time.clock()
        detections = self.detector.detect(image)
        print("detect time = {:.4f}".format(time.clock() - tic))
        return detections