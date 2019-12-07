from SiamMask.tools.test import *

class SiammaskTracker:
    def __init__(self, base_dir='', x=0 ,y=0,w=10,h=10,sample_im='SiamMask/data/tennis/00000.jpg',fp16_mode=True,features_trt=True,rpn_trt=False,mask_trt=False,refine_trt=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True

        # Setup Model
        args = argparse.Namespace()
        args.config = base_dir + 'SiamMask/experiments/siammask_sharp/config_davis.json'
        args.resume = base_dir + 'SiamMask/experiments/siammask_sharp/SiamMask_DAVIS.pth'
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
        im = cv2.imread(sample_im)
        self.state = siamese_init(im, target_pos, target_sz, siammask, self.cfg['hp'], device=self.device)  # init tracker
        self.state['net'].init_trt(fp16_mode,features_trt,rpn_trt,mask_trt,refine_trt)


    def track(self,im):
        self.state = siamese_track(self.state, im, mask_enable=True, refine_enable=True, device=self.device)  # track
        location = self.state['ploygon'].flatten()
        mask = self.state['mask'] > self.state['p'].seg_thr
        return location,mask

    def reinit(self,new_box,im):
        # new_box format: x,y,w,h (where x,y correspond to the top left corner)
        (x,y,w,h) = new_box
        target_pos = np.array([x + w / 2, y + h / 2])
        target_sz = np.array([w, h])
        self.state = siamese_init(im, target_pos, target_sz, self.state['net'], self.cfg['hp'], device=self.device)  # init tracker
