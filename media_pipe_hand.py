import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "/Users/wencysuo/code/cut-the-rope-tv/hand_landmarker_task.task"


def print_result(
    result: mp.tasks.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int
):
    print("hand landmarker result: {}".format(result))


options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions,
    running_mode=mp.tasks.vision.RunningMode,
    result_callback=print_result,
    num_hands=2,
)
with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
    # init landmarker so we can use it here

    # use opencv video capture to start capturing from webcam
    # read using default camera so input = 0
    cap = cv.VideoCapture(0)
    if not cap.isOpened:
        print("Camera failed")
        exit(0)

    # create loop to read the lastest frmae fromr the camera using videocapture read()
    while True:
        ret, frame = cap.read()
        if frame is None:
            print("failed to capture frames")
            break

        # call hand detection function here

        if cv.waitKey(10) == 27:
            break
