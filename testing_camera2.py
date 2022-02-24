import cv2
print ("starting grab")
cap = cv2.VideoCapture('http://localhost:8081/stream/video.mjpeg')
while True:
    ret, frame = cap.read()
    #print ("found frame")
    cv2.imshow('Video', frame)
    #cv2.imwrite("test.jpg",frame)
    #print ("done")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    #print ("after release")