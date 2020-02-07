from detector import YoloDetector
from tracker import SiammaskTracker
import sys, os, time
import numpy as np

class TrackedObject:
    def __init__(self, object_id, class_id):
        self.id = object_id
        self.active = True
        self.last_bb = None
        self.class_id = class_id
        self.latest_tracked_state = None

class ImageSegmentor:
    def __init__(self,sample_im,detector_name='yolov3',tracker_name='siammask', detect_class_ids=[80],use_trt=False, im_width=640, im_height=480):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/front_end/'
        if detector_name == 'yolov3':
            self.detector = YoloDetector(sample_im,base_dir=base_dir, classes_ids=detect_class_ids)
        else:
            raise RuntimeError("Detector chosen not implemented")

        if tracker_name == 'siammask':
            self.tracker = SiammaskTracker(sample_im,base_dir=base_dir, use_tensorrt=use_trt)
        else:
            raise RuntimeError("Tracker chosen not implemented")

        self.active_objects_ids_per_class = {}
        self.tracked_objects = []
        # self.last_boxes = []
        # self.last_classes = []
        self.ukf_dict = {}

        ####################################################################

        # Statistics used for testing new measurements
        self.z_090_one_sided = 1.282
        self.z_075_one_sided = 0.674
        self.z_050_one_sided = 0.0

        self.min_pix_from_edge = 5
        self.min_aspect_ratio = 1
        self.max_aspect_ratio = 5

        self.F_005 = 161.4476
        self.im_width = im_width
        self.im_height = im_height
        
    def track(self,image):
        tic = time.time()
        output = []
        # Go over each active tracked object
        for obj_id in sum(self.active_objects_ids_per_class.values(),[]):
            self.tracked_objects[obj_id].latest_tracked_state, abb, mask = self.tracker.track(image,self.tracked_objects[obj_id].latest_tracked_state)

            # Check if measurement valid if we have a state estimate
            if obj_id in self.ukf_dict:
                valid = self.valid_tracking(self.ukf_dict[obj_id],abb)
            else:
                valid = True

            output.append((abb,self.tracked_objects[obj_id].class_id,obj_id, valid))

        # dummy_object_ids = list(range(len(self.last_boxes)))  # DEBUG
        tic2 = time.time()
        print("track time = {:.4f}".format(tic2- tic))
        return output

    def new_tracked_object(self,class_id):
        obj_id = len(self.tracked_objects)
        self.tracked_objects.append(TrackedObject(obj_id,class_id))
        return obj_id


    def reinit_tracker(self,new_boxes,image):
        new_active_objects_ids_per_class = {}
        new_active_objects_ids = []
        for new_box in new_boxes:
            class_id = new_box[-1]
            if class_id not in self.active_objects_ids_per_class or len(self.active_objects_ids_per_class[class_id]) == 0:
                # No active objects of this class
                obj_id = self.new_tracked_object(class_id)
            else:
                # There exist some active objects of this class, check if they match
                best_t = self.F_005
                obj_id = None
                # Go through active candidate objects
                for id in self.active_objects_ids_per_class[class_id]:
                    # Make sure the candidate object isn't already matched to a new detection
                    if id in new_active_objects_ids:
                        continue
                    # Probabilistic check of if the measurement matches the ukf prediction
                    abb = np.concatenate([new_box[:4],[0]])
                    t = self.score_state_consistent_measurement(self.ukf_dict[id],abb)
                
                    # Found an acceptable match
                    if t < best_t:
                        best_t = t
                        obj_id = id

                # No object was matched, new object detected
                if obj_id == None:
                    obj_id = self.new_tracked_object(class_id)

            # Get tracker initialization from the detected box
            self.tracked_objects[obj_id].latest_tracked_state = self.tracker.reinit(new_box,image)
            if class_id in new_active_objects_ids_per_class:
                new_active_objects_ids_per_class[class_id].append(obj_id)
            else:
                new_active_objects_ids_per_class[class_id] = [obj_id]
            new_active_objects_ids.append(obj_id)

        # Remove from active objects the ones that have not been redetected
        self.active_objects_ids_per_class = new_active_objects_ids_per_class

    def detect(self,image):
        tic = time.time()
        detections = self.detector.detect(image)
        valid_detections_ids = []
        # Ignore invalid detections
        for i,detection in enumerate(detections):
            if self.valid_detection(detection):
                valid_detections_ids.append(i)
        detections = detections[valid_detections_ids,:]

        print("detect time = {:.4f}".format(time.time() - tic))
        return detections


    def score_state_consistent_measurement(self,ukf,abb):
        # Check if new measurement is too far from distribution of previous measurement
        # Hotelling's t-squared statistic
        return math.sqrt((abb-ukf.mu_obs)@ la.inv(ukf.S_obs) @ (abb-ukf.mu_obs).T)
        


    def valid_detection(self, detection):
        return True
        bb = detection[:4]
        class_id = detection[-1]
        # Left side in image
        if (bb[0]) < self.min_pix_from_edge:
            return False
        
        # Right side in image
        if (bb[0] + bb[2]) > self.im_width - self.min_pix_from_edge:
            return False

        # Top in image
        if (bb[1] ) < self.min_pix_from_edge:
            return False
        
        # Bottom in image
        if (bb[1] + bb[3]) > self.im_height - self.min_pix_from_edge:
            return False

        # Aspect ratio valid
        ar = bb[2]/bb[3]
        if ar < self.min_aspect_ratio or ar > self.max_aspect_ratio:
            print("rejecting measurement: INVALID ASPECT RATIO")
            return False

        return True


    def valid_tracking(self,ukf,abb):
        if self.mu_obs is None:
            return True
        # Check if row or column valid
        mu_x_l = abb[0] - abb[2]/2
        mu_x_r = abb[0] + abb[2]/2
        sigma_x = math.sqrt(ukf.S_obs[0,0] + ukf.S_obs[2,2]/4)
        z_x_l = (0-mu_x_l)/sigma_x
        z_x_r = (self.im_width-mu_x_r)/sigma_x

        if z_x_l > -self.z_075_one_sided or z_x_r < self.z_075_one_sided:
            rospy.loginfo("Rejected measurement with values left {} and right {}".format(z_x_l,z_x_r))
            return False

        mu_y_l = abb[1] - abb[3]/2
        mu_y_r = abb[1] + abb[3]/2
        sigma_y = math.sqrt(self.S_obs[1,1]+ self.S_obs[3,3]/4)
        z_y_l = (0-mu_y_l)/sigma_y
        z_y_r = (self.im_height-mu_y_r)/sigma_y

        if z_y_l > -self.z_075_one_sided or z_y_r < self.z_075_one_sided:
            rospy.loginfo("Rejected measurement with values top {} and bottom {}".format(z_y_l,z_y_r))
            return False


        t = self.score_state_consistent_measurement(ukf,abb)
        if t > self.F_005:
            rospy.loginfo("Rejected measurement too far from distribution: F={}".format(t))
            return False

        return True