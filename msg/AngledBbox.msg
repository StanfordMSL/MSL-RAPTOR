# This custom message is a MSL-RAPTOR angled bounding box
Header header
float64 x  # x coordinate of center of bounding box
float64 y  # y coordinate of center of bounding box
float64 width  # width (long axis) of bounding box
float64 height  # height (short axis) of bounding box
float64 angle  # angle in radians
uint8 im_seg_mode  # self.DETECT = 1  self.TRACK = 2  self.FAKED_BB = 3  self.IGNORE = 4
string class_str
uint8 id