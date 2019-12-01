from detector import YoloDetector
from tracker import SiammaskTracker

class ImageSegmentor:
    def __init__(self,detector_name='yolov3',tracker_name='siammask'):
        if detector_name == 'yolov3':
            self.detector = YoloDetector()
        else:
            raise RuntimeError("Detector chosen not implemented")

        if tracker_name == 'siammask':
            self.tracker = SiammaskTracker()
        else:
            raise RuntimeError("Tracker chosen not implemented")

        self.last_box = None
        
    def track(self,image):
        self.last_box,_ = self.tracker.track(image)
        return self.last_box

    def reinit_tracker(self,new_box,image):
        self.tracker.reinit(new_box,image)

    def detect(self,image):
        detections = self.detector.detect(image)
        return detections