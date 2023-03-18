from __future__ import print_function
import cv2 as cv
import argparse
import playsound as ps
from threading import Thread

REQUIRED_DETECTIONS = 10


def detect_and_display(frame, bird_cascade):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    birds = bird_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in birds:
        center = (x + w // 2, y + h // 2)
        frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)

    font = cv.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    font_scale = 0.6
    color = (255, 0, 0)
    thickness = 1
    detected = len(birds) > 0
    if detected:
        frame = cv.putText(frame, 'Detected Bird!', org, font, font_scale, color, thickness, cv.LINE_AA)
    cv.imshow('Capture - Bird detection', frame)
    return detected


def main():
    # get arguments
    parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
    parser.add_argument('--bird_cascade', help='Path to face cascade.',
                        default='data/haarcascades/haarcascade_bird.xml')
    parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
    args = parser.parse_args()
    bird_cascade_name = args.bird_cascade
    bird_cascade = cv.CascadeClassifier()

    # Load the cascades
    if not bird_cascade.load(cv.samples.findFile(bird_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)
    camera_device = args.camera
    # Read the video stream
    cap = cv.VideoCapture(camera_device)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)

    num_consecutive_detections = 0
    # Detection loop
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        detected = detect_and_display(frame, bird_cascade)
        if detected:
            num_consecutive_detections += 1
            if num_consecutive_detections >= REQUIRED_DETECTIONS:
                Thread(target=ps.playsound, args=('data/sounds/hawk-scream.mp3',)).start()
                num_consecutive_detections = 0
        else:
            num_consecutive_detections = 0
        if cv.waitKey(10) == 27:
            break


if __name__ == '__main__':
    main()
