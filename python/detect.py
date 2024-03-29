from ultralytics import YOLO 
import time
import datetime
import os
import sys
import csv
import argparse

videoname = 'video/abc.mp4'
save_video = False

parser = argparse.ArgumentParser(description='Detect birds from a video.')
parser.add_argument('videofilename', help='Name of the video file', nargs='?', default=videoname)
args = parser.parse_args()

os.chdir('yolo')
dirbase = './'
detection_model = YOLO('model/best.pt')
source_path = args.videofilename
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


csv1 = open('results1.csv', 'w', encoding='utf-8', newline='')
csv2 = open('results2.csv', 'w', encoding='utf-8', newline='')
resfile = open('all-results.txt', 'w', encoding='utf-8', newline='')

cwriter1 = csv.writer(csv1)
cwriter2 = csv.writer(csv2)
c = 0
for rone in results:
    c += 1
    cstr = '++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n' + 'c = ' + str(c) + '\n'
    resfile.write(cstr)
    for object in rone.boxes:
        xywh = object.xywhn.cpu().numpy()
        x = xywh[0,0]; y = xywh[0,1]
        w = xywh[0,2]; h = xywh[0,3]
        xyxy = object.xyxyn.cpu().numpy()
        x1 = xyxy[0,0]; y1 = xyxy[0,1]
        x2 = xyxy[0,2]; y2 = xyxy[0,3]
        if object.id != None:
            id = int(object.id.cpu().numpy()[0])
        else:
            id = 0
        rstr = str(object) + '\n'
        resfile.write(rstr)
        ob = [c, id, x, y, w, h]
        cwriter1.writerow(ob)
        ob = [c, id, x1, y1, x2, y2]
        cwriter2.writerow(ob)


csv1.close()
csv2.close()
resfile.close()

time2 = time.time()
dt_end = datetime.datetime.now()
dtime = time2 - time1

print('\nStart Time: ' + str(dt_start))
print(  'End   Time: ' + str(dt_end))
print(  'Exec  Time: ' + str(dtime) + ' sec')

exit(0)

# with open('results90.csv', 'w', encoding='utf-8', newline='') as cfile:
#     cwriter = csv.writer(cfile)
#     c = 0;
#     for rone in results:
#         c += 1
#         for object in rone.boxes:
#             #print("object.numpy():")
#             #print(object.xywh.cpu().numpy())
#             xywh = object.xywhn.cpu().numpy()
#             x = xywh[0,0]
#             y = xywh[0,1]
#             w = xywh[0,2]
#             h = xywh[0,3]
#             if object.id != None:
#                 id = int(object.id.cpu().numpy()[0])
#             else:
#                 id = 0
#             ob = [c, id, x, y, w, h]
#             cwriter.writerow(ob)

# with open('results92.csv', 'w', encoding='utf-8', newline='') as cfile:
#     cwriter = csv.writer(cfile)
#     c = 0;
#     for rone in results:
#         c += 1
#         for object in rone.boxes:
#             xyxy = object.xyxyn.cpu().numpy()
#             x1 = xyxy[0,0]
#             y1 = xyxy[0,1]
#             x2 = xyxy[0,2]
#             y2 = xyxy[0,3]
#             if object.id != None:
#                 id = int(object.id.cpu().numpy()[0])
#             else:
#                 id = int(0)
#             ob = [c, id, x1, y1, x2, y2]
#             cwriter.writerow(ob)

with open('results.txt', 'w', encoding='utf-8', newline='') as tfile:
    c = 0;
    for rone in results:
        c += 1
        cstr = '++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n' + 'c = ' + str(c) + '\n'
        tfile.write(cstr)
        for object in rone.boxes:
            rstr = str(object) + '\n'
            tfile.write(rstr)
