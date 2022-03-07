"""Fast tracking of a QR code in a video with the use of cropping.

Rough explanation
-----------------

- Read video frame
- Estimate new position of QR code
- Search for QR code near this position
- If QR code found,
    - update position estimate for next frame
- If no QR code found for more than 5x in a row,
    - search entire image for QR code
"""

import cv2
import sys
sys.path.insert(1, "cv") 
# Ignore these import errors
from find_qr import *

video_file = 'test_vids/qr2.mp4'
DIM = (np.array((1920, 1080))).astype(int)
USE_CROP = True # Whether to crop image around an estimated QR code area
TRACKER = False  # True = ROI tracker, False = QR tracker
CROP_RADIUS = 200 # radius around last known point for cropping, used when TRACKER = False


if __name__ == '__main__':
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking", int(DIM[0]/1.5), int(DIM[1]/1.5))
    if TRACKER:
        tracker_types = ['KCF', 'CSRT']
        tracker_index = 1

        tracker_type = tracker_types[tracker_index]
        tracker_funcs = [cv2.TrackerKCF_create, cv2.TrackerCSRT_create]
        tracker = tracker_funcs[tracker_index]()

        print("Selected tracker type", tracker_type)
    else:
        lastCentre = np.zeros(2).astype(int)
        track_timeout = 5  # How many frames of consecutive tracking failure before resetting lastCentre
        track_fail_count = track_timeout # Always reset on first frame

    # Read video
    video = cv2.VideoCapture(video_file)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    qrDecoder = cv2.QRCodeDetector()

    if USE_CROP and TRACKER:
        roi = cv2.selectROI("Tracking", frame, False)
        # Initialize tracker with first frame and bounding box
        ok = tracker.init(frame, roi)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        transform = [0,0] # Transformation from cropped coords to real coords
        if USE_CROP:
            if TRACKER: # ROI tracker
                ok, roi = tracker.update(frame)

                # Draw bounding box
                if ok:
                    # Tracking success
                    p1 = (int(roi[0]), int(roi[1]))
                    p2 = (int(roi[0] + roi[2]), int(roi[1] + roi[3]))
                    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                    qrframe = frame[p1[1]:p2[1],p1[0]:p2[0]]
                    transform = p1
                else:
                    # Tracking failure
                    cv2.putText(frame, "Tracking failure detected", (100, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            else: # QR code tracker
                if track_fail_count < track_timeout:
                    qrframe = frame[lastCentre[1] - CROP_RADIUS:lastCentre[1] + CROP_RADIUS,
                                    lastCentre[0] - CROP_RADIUS:lastCentre[0] + CROP_RADIUS]
                    transform = [lastCentre[0] - CROP_RADIUS, lastCentre[1] - CROP_RADIUS]
                else:
                    qrframe = frame
            # Search for QR code in (potentially cropped) frame
            found, bbox = qrDecoder.detect(qrframe)
        else: # USE_CROP = false
            found, bbox = qrDecoder.detect(frame)

        if found:
            bbox = bbox[0]  # bbox is always a unit length list, so just grab the first element
            # now bbox is a list of 4 vertices relative to cropped coords, so transform them
            for i in range(len(bbox)):
                bbox[i,0] += transform[0]
                bbox[i,1] += transform[1]
            
            # check validity
            shape_data = getQRShape(bbox)
            isValid = shape_data[0] < 300**2 and shape_data[0] > 60**2 and shape_data[1] > 0.98
            # text_data = getQRData(frame, bbox, qrDecoder)
            # isValid = text_data == "bit.ly/3tbqjqL"

            # Draw blue border if valid, green otherwise.
            drawMarkers(frame, bbox, (255, 0, 0) if isValid else (0, 255, 0))

            # update position estimate and failure counter
            if USE_CROP:
                if isValid:
                    if not TRACKER:
                        lastCentre = np.mean(bbox, axis=0).astype(int)
                    track_fail_count = 0
                else:
                    track_fail_count += 1
        else: # not found
            cv2.putText(frame, "QR code not detected", (100, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            track_fail_count += 1

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        # Display fails on frame
        cv2.putText(frame, "fails : " + str(int(track_fail_count)), (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        # Display result
        
        cv2.rectangle(frame, (1,1), (DIM[0] - 2, DIM[1] - 2), (0,0,255), 2)
        cv2.imshow("Tracking", frame)
        # cv2.imshow("Tracking", qrframe)

        # Exit if ESC pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):  # if press SPACE bar
            break

    video.release()
    cv2.destroyAllWindows()
