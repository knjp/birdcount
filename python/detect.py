from ultralytics import YOLO 
import time
import datetime
import os
import shutil
import sys
import csv
import argparse

videoname = 'video/abc.mp4'

parser = argparse.ArgumentParser(description='Detect birds from a video.')
parser.add_argument('videofilename', help='Name of the video file', nargs='?', default=videoname)
parser.add_argument('--save_video', help='Save the video file', action='store_true')
args = parser.parse_args()

os.chdir('yolo')
dirbase = './'
detection_model = YOLO('model/best.pt')
source_path = os.path.abspath(args.videofilename)
save_video = args.save_video

def saveVideoFile():
    SAVE_DIR = "outputs"
    VIDEO_DIR = "runs/detect/track"
    VIDEO_NAME = SAVE_DIR + '/output.avi'

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    if os.path.exists(VIDEO_NAME):
        os.remove(VIDEO_NAME)

    if not os.path.exists(VIDEO_DIR):
        exit(0)

    files = os.listdir(VIDEO_DIR)
    if len(files) == 1:
        videoFileName = VIDEO_DIR + '/' + files[0]
        new_filename = shutil.move(videoFileName, VIDEO_NAME)

    files = os.listdir(VIDEO_DIR)
    if len(files) == 0:
        os.rmdir(VIDEO_DIR)



time1 = time.time()
dt_start = datetime.datetime.now()
print('YOLO starts at ' + str(dt_start))
results = detection_model.track(source=source_path, save=save_video, 
                                #, device=0, save=False,
#                                conf=float(values['detection_conf_thres']),
#                                iou=float(values['detection_iou_thres']),
                                show=False,
                                conf=0.25,
                                iou=0.7,
                                save_txt=False,
                                save_frames=False,
                                persist=True,
                                stream=True,
                                verbose=False,
#                                save_crop=True
                                )

csv1 = open('results.csv', 'w', encoding='utf-8', newline='')
resfile = open('all-results.txt', 'w', encoding='utf-8', newline='')

cwriter1 = csv.writer(csv1)
#cwriter2 = csv.writer(csv2)
c = 0
#source_path = args.videofilename
infoVideoFile = "Video File: " + source_path
resfile.write(infoVideoFile + '\n')
csv1.write(infoVideoFile + '\n')

for rone in results:
    c += 1
    cstr = '++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n' + 'c = ' + str(c) + '\n'
    resfile.write(cstr)
    for object in rone.boxes:
        xywhn = object.xywhn.cpu().numpy()
        xn = xywhn[0,0]; yn = xywhn[0,1]
        wn = xywhn[0,2]; hn = xywhn[0,3]
        xyxyn = object.xyxyn.cpu().numpy()
        xn1 = xyxyn[0,0]; yn1 = xyxyn[0,1]
        xn2 = xyxyn[0,2]; yn2 = xyxyn[0,3]

        xyw = object.xywh.cpu().numpy()
        x = xyw[0,0]; y = xyw[0,1]
        w = xyw[0,2]; h = xyw[0,3]
        xyxy = object.xyxy.cpu().numpy()
        x1 = xyxy[0,0]; y1 = xyxy[0,1]
        x2 = xyxy[0,2]; y2 = xyxy[0,3]


        if object.id != None:
            id = int(object.id.cpu().numpy()[0])
        else:
            id = 0
        rstr = str(object) + '\n'
        resfile.write(rstr)
        ob = [c, id, x, y, w, h, x1, y1, x2, y2]
        cwriter1.writerow(ob)
        #ob = [c, id, xn, yn, wn, hn, xn1, yn1, xn2, yn2]
        #cwriter1.writerow(ob)


csv1.close()
resfile.close()

time2 = time.time()
dt_end = datetime.datetime.now()
dtime = time2 - time1

print('Start Time: ' + str(dt_start))
print(  'End   Time: ' + str(dt_end))
print(  'Exec  Time: ' + "{:.3f}".format(dtime) + ' sec')

if save_video:
    saveVideoFile()
