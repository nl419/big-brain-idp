"""Take screenshots of the camera video with w, quit with q"""

import cv2

folder = "dots"

print ("starting grab")
cap = cv2.VideoCapture('http://localhost:8081/stream/video.mjpeg')
prefix = "smol"
counter = 1
while True:
    ret, frame = cap.read()
    cv2.imshow('Video', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('w'):
        cv2.imwrite(folder + "/" + prefix + str(counter) + ".jpg", frame)
        counter += 1
cap.release()
cv2.destroyAllWindows()