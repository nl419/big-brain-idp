import cv2
import sys
from find_qr import *

# Credit: https://learnopencv.com/opencv-qr-code-scanner-c-and-python/
# Generate QR codes with https://barcode.tec-it.com/en/QRCode

# Obtain an affirmative lock (QRData is correct)
# Create a circular mask around the previous known center
# On next frame, search for qr code within that mask
# If tracking lost for 5 frames in a row, go to step 1

video_file = 'test_vids/qr2.mp4'
RESOLUTION = (np.array((1920, 1080)) / 2).astype(int)
TRACKER = False  # Whether to use an ROI tracking method

if __name__ == '__main__':
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking", RESOLUTION[0], RESOLUTION[1])
    if TRACKER:
        tracker_types = ['KCF', 'CSRT']
        tracker_index = 1

        tracker_type = tracker_types[tracker_index]
        tracker_funcs = [cv2.TrackerKCF_create, cv2.TrackerCSRT_create]
        tracker = tracker_funcs[tracker_index]()

        print("Selected tracker type", tracker_type)
    else:
        lastCentre = None
        track_timeout = 5  # How many frames of consecutive tracking failure before resetting lastCentre
        track_fail_count = 0

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

    if TRACKER:
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

        if TRACKER:
            ok, roi = tracker.update(frame)

            # Draw bounding box
            if ok:
                # Tracking success
                p1 = (int(roi[0]), int(roi[1]))
                p2 = (int(roi[0] + roi[2]), int(roi[1] + roi[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            else:
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            qrframe = frame
        elif track_fail_count < track_timeout:
            # https://stackoverflow.com/questions/42004652/how-can-i-find-contours-inside-roi-using-opencv-and-python
            black = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
            cv2.circle(black, lastCentre, 200, (255,255,255), -1)
            gray = cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)
            ret, b_mask = cv2.threshold(gray, 127, 255, 0)
            qrframe = cv2.bitwise_and(frame, frame, mask=b_mask)
        else:
            qrframe = frame

        # Detect the qrcode
        found, bbox = qrDecoder.detect(frame)
        if found:
            bbox = bbox[0]  # bbox is always a unit length list, so just grab the first element
            shape_data = getQRShape(bbox)
            isValid = shape_data[0] < 300**2 and shape_data[0] > 60**2 and shape_data[1] > 0.98
            # text_data = getQRData(frame, bbox, qrDecoder)
            # isValid = text_data == "bit.ly/3tbqjqL"

            # Draw blue border if valid, green otherwise.
            drawMarkers(frame, bbox, (255, 0, 0) if isValid else (0, 255, 0))
            if not TRACKER:
                if isValid:
                    lastCentre = np.mean(bbox, axis=0).astype(int)
                    track_fail_count = 0
                else:
                    track_fail_count += 1
        else:
            cv2.putText(frame, "QR code not detected", (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # if press SPACE bar
            break

    video.release()
    cv2.destroyAllWindows()
