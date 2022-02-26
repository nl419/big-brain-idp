from turtle import color
import cv2
import numpy as np

# Credit: https://learnopencv.com/opencv-qr-code-scanner-c-and-python/
# Generate QR codes with https://barcode.tec-it.com/en/QRCode

DECODE = False # Whether to additionally decode the QR code

img = cv2.imread("qr_codes/small/rotated.png")
# img = cv2.imread("qr_codes/small/original.png")

# Display barcode and QR code location
def display(im, bbox):
    bbox = bbox[0].astype(int)
    n = len(bbox)
    for j in range(n):
        p1 = tuple(bbox[j])
        p2 = tuple(bbox[(j+1) % n])
        cv2.line(im, p1, p2, (255,0,0), 3)

    centre = np.mean(bbox, axis=0).astype(int)
    top_midpoint = np.mean(bbox[0:2], axis=0).astype(int)
    marker_radius = 5
    print(centre)
    print(top_midpoint)
    cv2.circle(im, centre, radius=marker_radius, color=(0,0,255), thickness=-1)
    cv2.circle(im, top_midpoint, radius=marker_radius, color=(0,0,255), thickness=-1)
    cv2.line(im, centre, top_midpoint, color=(0,0,255), thickness=3)


    # Display results
    cv2.imshow("Results", im)

qrDecoder = cv2.QRCodeDetector()

if DECODE:
    # Detect and decode the qrcode
    data,bbox,rectifiedImage = qrDecoder.detectAndDecode(img)
    if len(data)>0:
        print("Decoded Data: {}".format(data))
        display(img, bbox)
    else:
        print("QR Code not detected")
        cv2.imshow("Results", img)
else:
    # Detect the qrcode
    found,bbox = qrDecoder.detect(img)
    if found:
        print("Found QR code")
        print(bbox)
        display(img, bbox)
    else:
        print("QR Code not detected")
        cv2.imshow("Results", img)

cv2.waitKey(0)
cv2.destroyAllWindows()