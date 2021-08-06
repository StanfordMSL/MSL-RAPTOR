from typing import DefaultDict
from detector import YoloDetector
from detector import EdgeTPU
from tracker import SiammaskTracker
import sys, os, time, pdb
import numpy as np
import math
import numpy.linalg as la
from utils_msl_raptor.ukf_utils import bb_corners_to_angled_bb, condensed_to_square, state_to_tf
from scipy.spatial.distance import pdist,squareform
class TrackedObject:
    def __init__(self, object_id, class_str):
        self.id = object_id
        self.class_str = class_str
        self.latest_tracked_state = None

class ImageSegmentor:
    def __init__(self,sample_im,detector_name='yolov3',tracker_name='siammask', detect_classes_ids=[0,39,41,45,63,80], detect_classes_names = ['person','bottle','cup','bowl','laptop','mslquad'],use_trt=False, im_width=640, im_height=480, detection_period = 5,verbose=False, use_track_checks=True, use_gt_detect_bb=False, detector_cfg='yolov3/cfg/yolov3.cfg', detector_weights='yolov3/weights/yolov3.weights'):
        self.last_object_id = -1
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/front_end/'
        print('Using classes '+str(detect_classes_names))
        if detector_name == 'yolov3':
            self.detector = YoloDetector(sample_im, base_dir=base_dir, classes_ids=detect_classes_ids, cfg=detector_cfg, weights=detector_weights)
        if detector_name == 'edge_tpu_mobile_det':
            self.detector = EdgeTPU(sample_im, classes_ids=detect_classes_ids)
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
        self.max_aspect_ratio = {'person':0.4,'mslquad':5,'bowl':2.5,'cup':1.3,'laptop':3,'bottle':1.1}

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

        self.obj_id_last_active_times_by_class = {} #DefaultDict(dict)  # [my_class_str][my_obj_id] = time_last_seen
        self.stale_detect_time_thresh = 5  # seconds since we last detected an object beyond which we dont try to match to it again and call it new
        self.trans_diff_thresh_for_bb_match = 0.2 # [m] distance in space for which we consider a new detection a match

    def stop_tracking_lost_objects(self):
        # Remove objects that triggered detection and were not matched to new detections
        self.last_lost_objects = set(self.last_lost_objects)
        for obj_id in self.last_lost_objects:
            c = self.tracked_objects[obj_id].class_str
            print('Removing '+c)
            self.active_objects_ids_per_class[c].remove(obj_id)
        self.last_lost_objects = []


    def process_image(self,image,time,gt_boxes=None, b_detect_only=False, tf_w_ego=None):
        '''
        Process an image by running detection and/or tracking and returning the bouding box, according to the state of the image segmentor.
        The gt_boxes is an optional argument which is used when using ground-truth for the detected boxes
        gt_boxes format: list of tuples: [(x,y,w,h,class_conf,obj_conf,class_id),...] where x and y are top left corner positions.
        ''' 

        if b_detect_only:
            if self.use_gt_detect_bb:
                if gt_boxes is None:
                    RuntimeError('Trying to use groundtruth boxes for detection, but none were given')
                bbs_no_angle = gt_boxes
            else:
                bbs_no_angle = self.detect(image)  # returns a list of tuples: [(bb, class conf, object conf, class_id), ...]
            self.last_detection_time = time
            # No detections
            if len(bbs_no_angle) == 0:
                # print("Did not detect object")
                # self.stop_tracking_lost_objects()
                return {}
            
            # # Add buffer around detections
            # bbs_no_angle[:,2:4] += self.box_buffer
            # # Detections to reinit tracker
            # self.reinit_tracker(bbs_no_angle, image)
            # self.mode = self.TRACK
            # output = self.track(image)

            output = {}
            for bb in bbs_no_angle:
                if not self.valid_detection:
                    continue
                class_str = self.class_id_to_str[bb[6]]

                # to detect if its a new object, check objects in our list
                bb_as_abb = np.concatenate([bb[:4],[0]])  # pretent our bounding box is angled
                min_t_val = np.inf
                matched_id = None
                obj_id_to_rm = []
                new_potential_matches = []
                if class_str in self.obj_id_last_active_times_by_class and len(self.obj_id_last_active_times_by_class[class_str].keys()) > 0:
                    for obj_id_match_candidate in self.obj_id_last_active_times_by_class[class_str].keys():
                        if not obj_id_match_candidate in self.ukf_dict:
                            # if this is the second loop of this function on the first call, there could have been something added in the previous 
                            #     loop to the obj_id_last_active_times_by_class dict() that isnt yet in the ukf_dict - skip it
                            continue
                        if abs(time - self.obj_id_last_active_times_by_class[class_str][obj_id_match_candidate]) > self.stale_detect_time_thresh:
                            obj_id_to_rm.append(obj_id_match_candidate)
                            continue
                        t = self.compute_mahalanobis_dist(self.ukf_dict[obj_id_match_candidate], bb_as_abb)
                        if not hasattr(self.ukf_dict[obj_id_match_candidate], 'mu_obs'):
                            # this means we only just made the UKF (the update step hasnt been called yet). In this case do the check by translation only
                            guessed_candidate_pose = self.ukf_dict[obj_id_match_candidate].approx_pose_from_bb(bb, tf_w_ego)
                            new_potential_matches.append((obj_id_match_candidate, self.ukf_dict[obj_id_match_candidate], guessed_candidate_pose))
                        if t < 2*self.chi2_001 and t < min_t_val:
                            # if t is less than some probablistic threshold AND it is our closest match
                            min_t_val = t
                            matched_id = obj_id_match_candidate
                    
                    if min_t_val is np.inf and len(new_potential_matches) > 0:
                        min_t_diff = np.Inf
                        for (obj_id_match_candidate, ukf_cand, guessed_candidate_pose) in new_potential_matches:
                            t_w_ado_bb_guess = guessed_candidate_pose[0]
                            t_w_ado_ukf = state_to_tf(ukf_cand.mu)[0:3, 3]
                            trans_diff = la.norm(t_w_ado_ukf - t_w_ado_bb_guess)
                            if trans_diff < self.trans_diff_thresh_for_bb_match and trans_diff < min_t_val:
                                min_t_val = trans_diff
                                matched_id = obj_id_match_candidate
                        if matched_id is not None:
                            print("Matched via 3D space")


                    if len(obj_id_to_rm) > 0:
                        for id in obj_id_to_rm:
                            del self.obj_id_last_active_times_by_class[class_str][id] # remove this obj_id from our dict
                            if len(self.obj_id_last_active_times_by_class[class_str].keys()) == 0:
                                del self.obj_id_last_active_times_by_class[class_str]
                
                b_new_object = matched_id is None   
                if not b_new_object:
                    # print("Re-found object")
                    obj_id = matched_id
                else:
                    # print("found new object")
                    obj_id = self.last_object_id + 1

                if not class_str in self.obj_id_last_active_times_by_class:
                    self.obj_id_last_active_times_by_class[class_str] = {}
                self.obj_id_last_active_times_by_class[class_str][obj_id] = time

                self.last_object_id = obj_id
                output[obj_id] = [bb_as_abb, class_str, True]
                # output[obj_id] = [(bb[0], bb[1], bb[2], bb[3], 0), class_str, True]
            
            return output



            # output appears to be [((x,y,w,h,ang), class_str,b_is_valid)...]
            bbs_no_angle = self.detect(image) 
            # format as if angled bounding box (but 0 for angle always)
            # obj_id is the same if we think its the same object (need to know prior info for this). for now make it a new one each time?
            output = {}
            for bb in bbs_no_angle:
                if not self.valid_detection:
                    continue
                class_str = self.class_id_to_str[bb[6]]

                # loop through known objects to find matches
                for prev_obj_id in self.active_objects_ids_per_class[class_str]:
                    prev_pos= self.tracked_objects[prev_obj_id].latest_tracked_state['target_pos']
                    if prev_obj_id in self.ukf_dict:
                        b_is_new_object = not self.valid_tracking(self.ukf_dict[prev_obj_id], bb, self.tracked_objects[prev_obj_id].latest_tracked_state['score'], prev_obj_id)
                    else:  # If this is a new object it has already passed the detection check, it is considered valid for now
                        b_is_new_object = True
                    
                    if b_is_new_object:
                        obj_id = self.new_tracked_object(class_str)
                        self.active_objects_ids_per_class[class_str] = [obj_id]
                    else:
                        pass
                raise RuntimeError

                obj_id = 0
                if len(self.active_objects_ids_per_class) > 0: 
                    for obj_id_list in self.active_objects_ids_per_class:
                        obj_id += len(obj_id_list)
                if class_str in self.active_objects_ids_per_class:
                    self.active_objects_ids_per_class[class_str].append(obj_id)
                else:
                    self.active_objects_ids_per_class[class_str] = [obj_id]
                output[obj_id] = [(bb[0], bb[1], bb[2], bb[3], 0), class_str, True]
            return output
        else:
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
        # print("calling general detection function")
        detections = self.detector.detect(image)
        if len(detections) == 0:
            return detections
        valid_detections_ids = []
        # Ignore invalid detections
        for i,detection in enumerate(detections):
            if self.valid_detection(detection):
                valid_detections_ids.append(i)
        detections = detections[valid_detections_ids,:]

        # print("detect time = {:.4f}".format(time.time() - tic))
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
            # print('Rejected measurement, too close to the left edge')
            return False
        
        # Right side in image
        if (bb[0] + bb[2]) > self.im_width - self.min_pix_from_edge:
            # print('Rejected measurement, too close to the right edge')
            return False

        # Top in image
        if (bb[1] ) < self.min_pix_from_edge:
            # print('Rejected measurement, too close to the top edge')
            return False
        
        # Bottom in image
        if (bb[1] + bb[3]) > self.im_height - self.min_pix_from_edge:
            # print('Rejected measurement, too close to the bottom')
            return False

        # # Aspect ratio valid
        # ar = bb[2]/bb[3]
        # if ar < self.min_aspect_ratio[class_str] or ar > self.max_aspect_ratio[class_str]:
        #     print("rejecting measurement: INVALID ASPECT RATIO for "+class_str+": "+str(ar))
        #     return False

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


        
