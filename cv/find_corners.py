import cv2, queue, threading, time

CHESSBOARD = (3,3)

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()

cap = VideoCapture('http://localhost:8081/stream/video.mjpeg')
while True:
    #time.sleep(.5)   # simulate time between events
    frame = cap.read()

    # Basic corner detection
    # gray = cv2.cvtColor(frame, COLOR_BGR2GRAY)
    # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #                 cv2.THRESH_BINARY,401,2)
    # corners = cv2.cornerHarris(gray, 2,3,0.04)
    # corners = cv2.dilate(corners, None)
    # frame[corners>0.01*corners.max()] = [0,0,255]

    ret, corners = cv2.findChessboardCorners(frame, CHESSBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    frame = cv2.drawChessboardCorners(frame, CHESSBOARD, corners, ret)
    print(corners)
    cv2.imshow("frame", frame)
    if chr(cv2.waitKey(1)&255) == 'q':
        break