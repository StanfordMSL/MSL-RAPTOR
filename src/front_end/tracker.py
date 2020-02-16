from SiamMask.tools.test import *

class SiammaskTracker:
    def __init__(self,sample_im, base_dir='', x=0 ,y=0,w=10,h=10, use_tensorrt=False,fp16_mode=True,features_trt=True,rpn_trt=False,mask_trt=False,refine_trt=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True

        # Setup Model
        args = argparse.Namespace()
        args.config = base_dir + 'SiamMask/experiments/siammask_sharp/config_vot.json'
        args.resume = base_dir + 'SiamMask/experiments/siammask_sharp/SiamMask_VOT.pth'
        self.cfg = load_config(args)
        from custom import Custom
        siammask = Custom(anchors=self.cfg['anchors'])
        siammask = load_pretrain(siammask, args.resume)
        if args.resume:
            assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
            siammask = load_pretrain(siammask, args.resume)

        siammask.eval().to(self.device)
    
        target_pos = np.array([x + w / 2, y + h / 2])
        target_sz = np.array([w, h])
        self.state = siamese_init(sample_im, target_pos, target_sz, siammask, self.cfg['hp'], device=self.device)  # init tracker
        if use_tensorrt:
             self.state['net'].init_trt(fp16_mode,features_trt,rpn_trt,mask_trt,refine_trt, trt_weights_path='/root/msl_raptor_ws/src/msl_raptor/src/front_end/SiamMask/weights_trt')

        self.keys_to_share = ['target_pos','target_sz','score','mask','ploygon']

        self.states_each_object = []
        self.current_classes = []

    def track(self,im,obj_state):
        locations = []
        masks = []
        with torch.no_grad():
            self.object_state_to_siammask_state(obj_state)
            self.state = siamese_track(self.state, im, mask_enable=True, refine_enable=True, device=self.device)  # track
            obj_state = self.siammask_state_to_object_state()
        locations.append(self.state['ploygon'].flatten())
        masks.append(self.state['mask'] > self.state['p'].seg_thr)
        return obj_state,self.state['ploygon'].flatten(),self.state['mask'] > self.state['p'].seg_thr

    def reinit(self,new_box,im):
        # new_box format: x,y,w,h (where x,y correspond to the top left corner)
        # self.states_each_object = []
        # self.current_classes = []
        (x,y,w,h) = new_box[:4]
        target_pos = np.array([x + w / 2, y + h / 2])
        target_sz = np.array([w, h])
        self.state = siamese_init(im, target_pos, target_sz, self.state['net'], self.cfg['hp'], device=self.device)  # init tracker
        self.state['score'] = None
        self.state['ploygon'] = None
        self.state['mask'] = None
        return self.siammask_state_to_object_state()
        

    def object_state_to_siammask_state(self,object_state):
        for key in self.keys_to_share:
            self.state[key] = object_state[key]

    def siammask_state_to_object_state(self):
        object_state = {}
        for key in self.keys_to_share:
            object_state[key] = self.state[key]
        return object_state