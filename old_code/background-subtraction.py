# see https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html

from __future__ import print_function
import cv2 as cv
import numpy as np

video_file = "test_vids/input.mp4"
algos = ['MOG2', 'KNN']
algo_index = 0

denoising_type = 0  # 1 is very, very slow.

if algo_index == 0:
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()
capture = cv.VideoCapture(cv.samples.findFileOrKeep(video_file))
if not capture.isOpened():
    print('Unable to open: ' + video_file)
    exit(0)
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    # Denoising attempt 1
    if denoising_type == 0:
        N = 10
        its = 5
        kernel = np.ones((N,N),np.uint8)
        frame = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel, iterations=its)

    # Denoising attempt 2
    else:
        frame = cv.fastNlMeansDenoisingColored(frame, None, 2, 2, 7, 21)

    fgMask = backSub.apply(frame)
    
    
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break