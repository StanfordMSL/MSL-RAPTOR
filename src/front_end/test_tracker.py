import glob
from tracker import SiammaskTracker
from tools.test import *
img_files = sorted(glob.glob(join('SiamMask/data/tennis/*.jp*')))
ims = [cv2.imread(imf) for imf in img_files]
print(len(ims))

t = SiammaskTracker(fp16_mode=True,features_trt=True,rpn_trt=False,mask_trt=False,refine_trt=False)

cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
try:
    init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
    x, y, w, h = init_rect
except:
    exit()

toc = 0
for f, im in enumerate(ims):
        
    if f == 0:  # init
        t.reinit([x,y,w,h],im)
    elif f > 0:  # tracking
        tic = cv2.getTickCount()
        location, mask = t.track(im) 

        im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
        cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
        cv2.imshow('SiamMask', im)
        key = cv2.waitKey(1)
        if key > 0:
            break

        toc += cv2.getTickCount() - tic
toc /= cv2.getTickFrequency()
fps = f / toc
print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))