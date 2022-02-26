"""Removes buffering from VideoCapture, so the latest frame is always grabbed.

Examples
--------
Read a videostream, and have some long calculations to do on every frame,
but always stay in sync with video
>>> cap = VideoCapture('http://localhost:8081/stream/video.mjpeg')
... while True:
...  time.sleep(.5)   # simulate time between events
...  frame = cap.read()
...  cv2.imshow("frame", frame)
...  waitKey(1)
"""

import cv2, queue, threading, time

# Credit: https://stackoverflow.com/questions/43665208/how-to-get-the-latest-frame-from-capture-device-camera-in-opencv

# bufferless VideoCapture
class VideoCapture:
  """Removes buffering from VideoCapture, so the latest frame is always grabbed.

  Examples
  --------
  Read a videostream, and have some long calculations to do on every frame,
  but always stay in sync with video
  >>> cap = VideoCapture('http://localhost:8081/stream/video.mjpeg')
  ... while True:
  ...  time.sleep(.5)   # simulate time between events
  ...  frame = cap.read()
  ...  cv2.imshow("frame", frame)
  ...  waitKey(1)
  """
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

if __name__ == "__main__":
  cap = VideoCapture('http://localhost:8081/stream/video.mjpeg')
  while True:
    time.sleep(.5)   # simulate time between events
    frame = cap.read()
    cv2.imshow("frame", frame)
    if chr(cv2.waitKey(1)&255) == 'q':
      break