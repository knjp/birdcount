import os
import cv2

os.chdir('yolo')
video_path = 'video/abc.mp4'

dir_path = './'
basename = 'frame'
start_frame = 1
stop_frame = 2
step_frame = 1
ext = 'png'

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    exit(1)

os.makedirs(dir_path, exist_ok=True)
base_path = os.path.join(dir_path, basename)

digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

for n in range(start_frame, stop_frame, step_frame):

    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite('{}_{}.{}'.format( base_path, str(n).zfill(digit), ext), frame)
    else:
        exit(0)

####################
####################
src_img = cv2.imread('frame_001.png')
hsv_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV_FULL)
b = cv2.convertScaleAbs(src_img, alpha=0.4, beta=150)
cv2.imwrite('frame_001a.png', b)
