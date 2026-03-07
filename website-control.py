import time

import cv2 as cv
import mediapipe as mp
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# import our hand tracking file
pyautogui.PAUSE = 0

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

screen_width, screen_height = 0, 0
latest_gesture = "None"
latest_landmarks = None
first_drag = False
bottom_margin = .75
top_margin = .1


# get monitor specs to use and setup hand tracking software also
def setup():
    # setup video capture
    cap = cv.VideoCapture(0)

    model_path = (
        "/Users/jfileto/Desktop/cut-the-rope-tv/Gesture Recognition Task Guide.task"
    )

    global screen_width, screen_height
    # playing with full screen -> just use full monitor size
    screen_width, screen_height = pyautogui.size()

    # setup video gesture to be live
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=result_callback,
    )

    recognizer = GestureRecognizer.create_from_options(options)

    return cap, recognizer


last_gesture = None


def result_callback(
    result: mp.tasks.vision.GestureRecognizerResult,
    output_image: mp.Image,
    timestamp_ms: int,
):
    # in this we have to remember the last gesture
    # if it continues to be 👆 then we can continue dragging from prev point
    # open palm is nothing 🖐
    # press/popping is fist 👊
    # if not we start dragging from this point
    # realistically the gesture will never be NONE since user will have closed fist
    # we dont care about handedness, only gestures, and landmarks

    global latest_gesture, latest_landmarks, first_drag, top_margin, bottom_margin
    if not result.gestures:
        latest_gesture = "None"
        latest_landmarks = None
        return

    gestureName = result.gestures[0][0].category_name

    latest_gesture = result.gestures[0][0].category_name
    latest_landmarks = result.hand_landmarks[0]

    if gestureName == "Pointing_Up":
        # x and y derives from landmark position
        # duration should be 0 so its somewhat immediate
        # 1/(7/8 - 1/10)

        if result.hand_landmarks[-1][8].y < top_margin:
            return
        if result.hand_landmarks[-1][8].y > bottom_margin:
            return

        point_x, point_y = (
            screen_width * result.hand_landmarks[-1][8].x,
            (screen_height * (result.hand_landmarks[-1][8].y - top_margin)) * 1./(bottom_margin - top_margin),
        )
        if first_drag is False:
            first_drag = True
            pyautogui.mouseDown(point_x, point_y)

        pyautogui.moveTo(point_x, point_y, duration=0)
    elif gestureName != "Pointing_Up":
        if first_drag is True:
            pyautogui.mouseUp()
            first_drag = False

    if gestureName == "Closed_Fist":
        # want to click once
        if result.hand_landmarks[-1][8].y < top_margin:
            return
        if result.hand_landmarks[-1][8].y > bottom_margin:
            return
        point_x, point_y = (
            screen_width * result.hand_landmarks[-1][4].x,
             (screen_height * (result.hand_landmarks[-1][4].y - top_margin)) * 1./(bottom_margin - top_margin),
        )
        pyautogui.click(x=point_x, y=point_y, clicks=1, interval=0, button="left")


# we can use the following gesture 👆 from
# https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/python


# control mouse movements
def main():
    cap, recognizer = setup()

    start_time = time.time()
    timestamp = 0
    num_frames = 0

    mp_drawing = mp.tasks.vision.drawing_utils
    mp_drawing_styles = mp.tasks.vision.drawing_styles
    mp_hands = mp.tasks.vision.HandLandmarksConnections

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("failed to capture frame")
            break
        num_frames += 1
        timestamp += 1

        # args: image, top left corner, bottom right corner,color, thickness
        # convert image to mp format
        frame = cv.flip(frame, 1)
        # resize the image
        frame = cv.resize(frame, (1280, 720))

        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        recognizer.recognize_async(mp_image, timestamp)

        # draw overlay in main thread
        display = frame.copy()
        h, w, _ = display.shape

        mp_drawing.draw_landmarks(
            display,
            latest_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

        cv.rectangle(display, (0, int(h * top_margin)), (w, int(h * bottom_margin)), (0, 0, 255), 2)
        cv.putText(display, latest_gesture, (20, int(h * top_margin + 30)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv.imshow("Hand Gesture Recognition with MediaPipe", display)
        # use esc key to close application
        if cv.waitKey(10) == 27:
            break

    cap.release()


if __name__ == "__main__":
    main()
