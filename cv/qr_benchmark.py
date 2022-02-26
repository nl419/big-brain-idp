"""Benchmark the time taken to find a QR code"""

from timeit import timeit

# Generate QR codes with https://barcode.tec-it.com/en/QRCode
# Achieved ~140 FPS on my machine

N = 100

setupString = '''
import cv2
img = cv2.imread("qr_codes/small/rotated.png")
qrDecoder = cv2.QRCodeDetector()
'''

runString = "qrDecoder.detect(img)"
duration = timeit(runString, setupString, number=N)/N

print("Average time per decode:", duration, "s")
print("=> Max FPS of", 1/duration)