# Haar Cascade Classifier Tutorial

import argparse

import cv2 as cv


def detectAndDisplay(frame):
    # process frame -> grayscale
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    # face detection
    faces = face_cascade.detectMultiScale(frame_gray)
    for x, y, w, h in faces:
        center = (x + w // 2, y + h // 2)
        frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)

        # face box?
        face_ROI = frame_gray[y : y + h, x : x + w]
        eyes = eyes_cascade.detectMultiScale(face_ROI)
        for x2, y2, w2, h2 in eyes:
            # drawing eyes as circle instead of elipse
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0), 4)

        cv.imshow("Capture - Face detection", frame)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--face_cascade", help="Path to face cascade", default="cascades/frontal_face.xml"
)
parser.add_argument(
    "--eyes_cascade", help="Path to eyes cascade", default="cascades/eye_tree.xml"
)
args = parser.parse_args()


face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade

face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()

if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print("Face Cascade load failed")
    exit(0)

if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print("Eye Cascade load failed")
    exit(0)

# read input using default camera
cap = cv.VideoCapture(0)
if not cap.isOpened:
    print("Camera failed")
    exit(0)

# main loop
while True:
    ret, frame = cap.read()
    if frame is None:
        print("failed to capture frames")
        break

    detectAndDisplay(frame)

    if cv.waitKey(10) == 27:
        break
