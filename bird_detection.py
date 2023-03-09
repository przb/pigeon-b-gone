from __future__ import print_function
import cv2 as cv
import argparse


def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    # -- Detect faces
    birds = bird_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in birds:
        center = (x + w // 2, y + h // 2)
        frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
        # -- In each face, detect eyes
    cv.imshow('Capture - Bird detection', frame)


parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--bird_cascade', help='Path to face cascade.',
                    default='data/haarcascades/haarcascade_bird.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
bird_cascade_name = args.bird_cascade
bird_cascade = cv.CascadeClassifier()
# -- 1. Load the cascades
if not bird_cascade.load(cv.samples.findFile(bird_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
camera_device = args.camera
# -- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    if cv.waitKey(10) == 27:
        break
