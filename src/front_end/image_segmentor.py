from detector import YoloDetector
from tracker import SiammaskTracker
import sys, os, time
import numpy as np
import math
import numpy.linalg as la
from utils_msl_raptor.ukf_utils import bb_corners_to_angled_bb
from utils_msl_raptor.ukf_utils import condensed_to_square
from scipy.spatial.distance import pdist,squareform
class TrackedObject:
    def __init__(self, object_id, class_str):
        self.id = object_id
        self.class_str = class_str
        self.latest_tracked_state = None

class ImageSegmentor:
    def __init__(self,sample_im,detector_name='yolov3',tracker_name='siammask', detect_classes_ids=[0,39,41,45,63,80], detect_classes_names = ['person','bottle','cup','bowl','laptop','mslquad'],use_trt=False, im_width=640, im_height=480, detection_period = 5,verbose=False, use_track_checks=True, use_gt_detect_bb=False):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/front_end/'
        print('Using classes '+str(detect_classes_names))
        if detector_name == 'yolov3':
            self.detector = YoloDetector(sample_im,base_dir=base_dir, classes_ids=detect_classes_ids)
        else:
            raise RuntimeError("Detector chosen not implemented")

        if tracker_name == 'siammask':
            self.tracker = SiammaskTracker(sample_im,base_dir=base_dir, use_tensorrt=use_trt)
        else:
            raise RuntimeError("Tracker chosen not implemented")


        self.class_id_to_str = dict(zip(detect_classes_ids, detect_classes_names))
        self.class_str_to_id = dict(zip(detect_classes_names,detect_classes_ids))

        self.active_objects_ids_per_class = {}
        self.tracked_objects = []
        self.last_lost_objects = []
        # self.last_boxes = []
        # self.last_classes = []
        self.ukf_dict = {}

        # Used to decide when to track and to detect
        self.DETECT = 1
        self.TRACK = 2

        self.mode = self.DETECT

        # Pixels added around the bounding box used to initialize tracker
        # self.box_buffer = -10
        self.box_buffer = 0

        self.last_detection_time = None
        self.detection_period = detection_period

        ####################################################################

        # Statistics used for testing new measurements
        self.z_l_00005 = -3.32
        self.z_r_00005 = 3.32

        self.z_09998_one_sided = 3.49
        self.z_099_one_sided = 2.33
        self.z_090_one_sided = 1.282
        self.z_075_one_sided = 0.674
        self.z_050_one_sided = 0.0

        self.min_pix_from_edge = 5
        # Dicts containing each class
        self.min_aspect_ratio = {'person':0.1,'mslquad':1,'bowl':0.5,'cup':0.2,'laptop':0.3,'bottle':0.1}
        self.max_aspect_ratio = {'person':0.4,'mslquad':5,'bowl':2,'cup':1.3,'laptop':3,'bottle':1.0}

        self.chi2_03 = 6.06
        self.chi2_005 = 11.07
        self.chi2_001 = 20.52


        self.im_width = im_width
        self.im_height = im_height

        self.track_min_score = 0.3
        self.min_square_pix_dist_other_objs = 16

        self.verbose = verbose
        self.use_track_checks = use_track_checks
        self.use_gt_detect_bb = use_gt_detect_bb
        
        self.num_detections = 0

    def stop_tracking_lost_objects(self):
        # Remove objects that triggered detection and were not matched to new detections
        self.last_lost_objects = set(self.last_lost_objects)
        for obj_id in self.last_lost_objects:
            c = self.tracked_objects[obj_id].class_str
            print('Removing '+c)
            self.active_objects_ids_per_class[c].remove(obj_id)
        self.last_lost_objects = []


    def process_image(self,image,time,gt_boxes=None):
        '''
        Process an image by running detection and/or tracking and returning the bouding box, according to the state of the image segmentor.
        The gt_boxes is an optional argument which is used when using ground-truth for the detected boxes
        gt_boxes format: list of tuples: [(x,y,w,h,class_conf,obj_conf,class_id),...] where x and y are top left corner positions.
        ''' 
        if self.mode == self.DETECT:
            if self.use_gt_detect_bb:
                if gt_boxes is None:
                    RuntimeError('Trying to use groundtruth boxes for detection, but none were given')
                bbs_no_angle = gt_boxes
            else:
                bbs_no_angle = self.detect(image)  # returns a list of tuples: [(bb, class conf, object conf, class_id), ...]
            self.last_detection_time = time
            # No detections
            if len(bbs_no_angle) == 0:
                print("Did not detect object")
                self.stop_tracking_lost_objects()
                return self.track(image)
            
            # Add buffer around detections
            bbs_no_angle[:,2:4] += self.box_buffer
            # Detections to reinit tracker
            self.reinit_tracker(bbs_no_angle, image)
            self.mode = self.TRACK
            output = self.track(image)
            
            return output
        elif self.mode == self.TRACK:
            if self.use_track_checks:
                self.check_periodic_detection(time)
            return self.track(image)

    def track(self,image):
        if self.use_track_checks:
            return self.track_with_checks(image)
        else:
            return self.track_without_checks(image)

    def track_without_checks(self,image):
        tic = time.time()
        output = {}
        # Go over each active tracked object
        for obj_id in sum(self.active_objects_ids_per_class.values(),[]):
            self.tracked_objects[obj_id].latest_tracked_state, abb, mask = self.tracker.track(image,self.tracked_objects[obj_id].latest_tracked_state)
            abb = bb_corners_to_angled_bb(abb.reshape(-1,2))
            
            output[obj_id] = [abb,self.tracked_objects[obj_id].class_str, True]

        tic2 = time.time()
        if self.verbose:
            print("track time = {:.4f}".format(tic2- tic))
        return output

    def track_with_checks(self,image):
        tic = time.time()
        output = {}
        prev_positions = []
        new_positions = []
        obj_ids = []
        # Go over each active tracked object
        for obj_id in sum(self.active_objects_ids_per_class.values(),[]):
            prev_pos= self.tracked_objects[obj_id].latest_tracked_state['target_pos']
            self.tracked_objects[obj_id].latest_tracked_state, abb, mask = self.tracker.track(image,self.tracked_objects[obj_id].latest_tracked_state)
            abb = bb_corners_to_angled_bb(abb.reshape(-1,2))
            # Check if measurement valid if we have a state estimate
            if obj_id in self.ukf_dict:
                valid = self.valid_tracking(self.ukf_dict[obj_id],abb,self.tracked_objects[obj_id].latest_tracked_state['score'],obj_id)
                # If not valid we switch to detection
                if not valid:
                    self.last_lost_objects.append(obj_id)
                    self.mode = self.DETECT
                    
            # If this is a new object it has already passed the detection check, it is considered valid for now
            else:
                valid = True

            # Keep track for further checks later
            if valid:
                obj_ids.append(obj_id)
                prev_positions.append(prev_pos)
                new_positions.append(self.tracked_objects[obj_id].latest_tracked_state['target_pos'])

            output[obj_id] = [abb,self.tracked_objects[obj_id].class_str, valid]

        # Checked if any objects collapsed to the same position during tracking
        if len(obj_ids) > 1:
            collapsed_objs_ids = self.find_collapsed_objects(obj_ids,prev_positions,new_positions)

            for c_id in collapsed_objs_ids:
                self.last_lost_objects.append(c_id)
                self.mode = self.DETECT
                # Set valid to false
                output[c_id][-1] = False
                print('Object '+str(c_id)+': tracked position is too close to another object')

        # dummy_object_ids = list(range(len(self.last_boxes)))  # DEBUG
        tic2 = time.time()
        if self.verbose:
            print("track time = {:.4f}".format(tic2- tic))
        return output

    def new_tracked_object(self,class_str):
        obj_id = len(self.tracked_objects)
        self.tracked_objects.append(TrackedObject(obj_id,class_str))
        return obj_id



    def reinit_tracker(self,new_boxes,image):
        new_active_objects_ids = []
        for new_box in new_boxes:
            class_str = self.class_id_to_str[new_box[-1]]
            if class_str not in self.active_objects_ids_per_class or len(self.active_objects_ids_per_class[class_str]) == 0:
                # No active objects of this class
                obj_id = self.new_tracked_object(class_str)
                self.active_objects_ids_per_class[class_str] = [obj_id]
            else:
                # There exist some active objects of this class, check if they match
                best_t = self.chi2_001
                obj_id = None
                # Go through active candidate objects
                for id in self.active_objects_ids_per_class[class_str]:
                    # Make sure the candidate object isn't already matched to a new detection, and already has a ukf prediction
                    if id in new_active_objects_ids or id not in self.ukf_dict:
                        continue
                    # Probabilistic check of if the measurement matches the ukf prediction
                    abb = np.concatenate([new_box[:4],[0]])
                    t = self.compute_mahalanobis_dist(self.ukf_dict[id],abb)
                
                    # Found an acceptable match
                    if t < best_t:
                        best_t = t
                        obj_id = id

                
                if obj_id == None:
                    # No object was matched, new object detected
                    obj_id = self.new_tracked_object(class_str)
                    if class_str in self.active_objects_ids_per_class:
                        self.active_objects_ids_per_class[class_str].append(obj_id)
                    else:
                        self.active_objects_ids_per_class[class_str] = [obj_id]
                else:
                    # Previously tracked object was matched, if it triggered redetection we can keep it
                    if obj_id in self.last_lost_objects: self.last_lost_objects.remove(obj_id)

            # Get tracker initialization from the detected box
            self.tracked_objects[obj_id].latest_tracked_state = self.tracker.reinit(new_box,image)

            # Keep track of new objects to not consider them for association
            new_active_objects_ids.append(obj_id)

        # Remove objects that triggered detection and were not matched to new detections
        self.stop_tracking_lost_objects()

        

    def detect(self,image):
        tic = time.time()
        detections = self.detector.detect(image)
        if len(detections) == 0:
            return detections
        valid_detections_ids = []
        # Ignore invalid detections
        for i,detection in enumerate(detections):
            if self.valid_detection(detection):
                valid_detections_ids.append(i)
        detections = detections[valid_detections_ids,:]

        print("detect time = {:.4f}".format(time.time() - tic))
        self.num_detections += 1
        return detections


    def compute_mahalanobis_dist(self,ukf,abb):
        # Check if new measurement is too far from distribution of previous measurement
        # Mahalanobis distance
        if not hasattr(ukf,'mu_obs'):
            return np.inf
        return math.sqrt((abb-ukf.mu_obs)@ la.inv(ukf.S_obs) @ (abb-ukf.mu_obs).T)
        


    def valid_detection(self, detection):
        bb = detection[:4]
        class_str = self.class_id_to_str[detection[-1]]
        # Left side in image
        if (bb[0]) < self.min_pix_from_edge:
            print('Rejected measurement, too close to the left edge')
            return False
        
        # Right side in image
        if (bb[0] + bb[2]) > self.im_width - self.min_pix_from_edge:
            print('Rejected measurement, too close to the right edge')
            return False

        # Top in image
        if (bb[1] ) < self.min_pix_from_edge:
            print('Rejected measurement, too close to the top edge')
            return False
        
        # Bottom in image
        if (bb[1] + bb[3]) > self.im_height - self.min_pix_from_edge:
            print('Rejected measurement, too close to the bottom')
            return False

        # Aspect ratio valid
        ar = bb[2]/bb[3]
        if ar < self.min_aspect_ratio[class_str] or ar > self.max_aspect_ratio[class_str]:
            print("rejecting measurement: INVALID ASPECT RATIO for "+class_str+": "+str(ar))
            return False

        return True


    def valid_tracking(self,ukf,abb,score,obj_id):
        if score < self.track_min_score:
            print('Object tracking score of '+str(score)+' - min '+str(self.track_min_score))
            return False

        if not hasattr(ukf,'mu_obs'):
            return True
        # Check if row or column valid
        mu_x_l = abb[0] - abb[2]/2
        mu_x_r = abb[0] + abb[2]/2
        sigma_x = math.sqrt(ukf.S_obs[0,0] + ukf.S_obs[2,2]/4)
        z_x_l = (0-mu_x_l)/sigma_x
        z_x_r = (self.im_width-mu_x_r)/sigma_x

        if z_x_l > -self.z_099_one_sided or z_x_r < self.z_099_one_sided:
            print("Rejected measurement with values left {} and right {} for {}".format(z_x_l,z_x_r,ukf.class_str))
            return False

        mu_y_l = abb[1] - abb[3]/2
        mu_y_r = abb[1] + abb[3]/2
        sigma_y = math.sqrt(ukf.S_obs[1,1]+ ukf.S_obs[3,3]/4)
        z_y_l = (0-mu_y_l)/sigma_y
        z_y_r = (self.im_height-mu_y_r)/sigma_y

        if z_y_l > -self.z_099_one_sided or z_y_r < self.z_099_one_sided:
            print("Rejected measurement with values top {} and bottom {}".format(z_y_l,z_y_r))
            return False


        t = self.compute_mahalanobis_dist(ukf,abb)
        if t > self.chi2_001:
            print("Rejected measurement too far from distribution: F={}".format(t))
            return False

        return True

    # If we haven't detected in too long, switch back to detection mode
    def check_periodic_detection(self,time):
        if time - self.last_detection_time > self.detection_period:
            self.mode = self.DETECT

    def find_collapsed_objects(self,obj_ids,prev_positions,new_positions):
        dists = pdist(new_positions, 'sqeuclidean')
        collapsed_ids = np.argwhere(dists < self.min_square_pix_dist_other_objs)

        obj_ids_collapsed = []
        # Check which of the positions changed most since its previous one and assume this is the wrong one
        for c_id in collapsed_ids:
            (i,j) = condensed_to_square(c_id,len(new_positions))
            if np.linalg.norm(new_positions[i]-prev_positions[i]) >  np.linalg.norm(new_positions[j]-prev_positions[j]):
                if obj_ids[i] not in obj_ids_collapsed: obj_ids_collapsed.append(obj_ids[i])
            else:
                if obj_ids[j] not in obj_ids_collapsed: obj_ids_collapsed.append(obj_ids[j])

        return obj_ids_collapsed


        
