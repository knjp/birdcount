import os
import moviepy.editor as moviepy

YOLO_OUTPUT =  'output.avi'
MP4FILE = 'output.mp4'
SAVE_DIR = 'yolo/outputs/'

aviFileName = SAVE_DIR + YOLO_OUTPUT
outputFileName = SAVE_DIR + MP4FILE

if os.path.exists(SAVE_DIR):
    clip = moviepy.VideoFileClip(aviFileName)
    clip.write_videofile(outputFileName)

