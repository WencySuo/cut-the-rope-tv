# %%
import datetime
import json
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, "./PyWebOSTV")

from pywebostv.connection import *
from pywebostv.controls import *
from pywebostv.discovery import *  # Because I'm lazy, don't do this.

# 1. For the first run, pass in an empty dictionary object. Empty store leads to an Authentication prompt on TV.
# 2. Go through the registration process. `store` gets populated in the process.
# 3. Persist the `store` state to disk.
# 4. For later runs, read your storage and restore the value of `store`.
load_dotenv()
TV_IP_ADDRESS = os.getenv("TV_IP_ADDRESS")


def storage_is_empty():
    try:
        with open("store.json", "r") as f:
            return False
    except FileNotFoundError:
        return True


def load_from_storage():
    with open("store.json", "r") as f:
        return json.load(f)


def setup_webos():
    if storage_is_empty():
        store = {}
    else:
        store = load_from_storage()
    client = WebOSClient(
        TV_IP_ADDRESS, secure=True
    )  # Use discover(secure=True) for newer models.
    client.connect()
    for status in client.register(store):
        if status == WebOSClient.PROMPTED:
            print("Please accept the connect on the TV!")
    with open("store.json", "w") as f:
        json.dump(store, f)

    return client


def launch_cut_the_rope(client):
    app = ApplicationControl(client)
    apps = app.list_apps()  # Returns a list of `Application` instances.

    # Let's launch cut the rope!
    cut_the_rope = [x for x in apps if "cut the rope" in x["title"].lower()][0]
    launch_info = app.launch(cut_the_rope)
    return launch_info


import time

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# import our hand tracking file

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# screenwidth and height cannot be fetched by session
# input custom dimensions here
screen_width, screen_height = 950 * 2, 540 * 2
latest_gesture = "None"
latest_landmarks = None
first_drag = False
bottom_margin = 0.75
top_margin = 0.1

cursor_x, cursor_y = 0, 0

# since we need to call client to write to webos
# it must be global for access in the callback function
# same for time_delta
client = None
inp = None
last_gesture_time = None


# get monitor specs to use and setup hand tracking software also
def setup():
    # setup video capture
    cap = cv.VideoCapture(0)

    model_path = (
        "/Users/jfileto/Desktop/cut-the-rope-tv/Gesture Recognition Task Guide.task"
    )

    # setup video gesture to be live
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=result_callback,
    )

    recognizer = GestureRecognizer.create_from_options(options)

    return cap, recognizer


last_gesture = None


def move_arb_distance(x, y, drag):
    # Determine the step direction (-1, 0, or 1)
    step_x = (x > 0) - (x < 0)
    step_y = (y > 0) - (y < 0)

    # Use absolute values for loop counts
    abs_x = abs(x)
    abs_y = abs(y)

    # Case 1: Pure Vertical Movement
    if x == 0 and y != 0:
        for _ in range(abs_y):
            inp.move(0, step_y, drag=drag)

    # Case 2: Pure Horizontal Movement
    elif y == 0 and x != 0:
        for _ in range(abs_x):
            inp.move(step_x, 0, drag=drag)

    # Case 3: Diagonal/Complex Movement
    elif x != 0 and y != 0:
        if abs_y >= abs_x:
            # Steep slope: Move Y multiple times for every 1 X
            slope = abs_y // abs_x
            for _ in range(abs_x):
                for _ in range(slope):
                    inp.move(0, step_y, drag=drag)
                inp.move(step_x, 0, drag=drag)

            # Handle remainder Y steps
            remainder = abs_y % abs_x
            for _ in range(remainder):
                inp.move(0, step_y, drag=drag)
        else:
            # Shallow slope: Move X multiple times for every 1 Y
            slope = abs_x // abs_y
            for _ in range(abs_y):
                for _ in range(slope):
                    inp.move(step_x, 0, drag=drag)
                inp.move(0, step_y, drag=drag)

            # Handle remainder X steps
            remainder = abs_x % abs_y
            for _ in range(remainder):
                inp.move(step_x, 0, drag=drag)

    # Case 4: Zero movement (Jitter/Ping)
    else:
        inp.move(1, 1, drag=drag)
        inp.move(-1, -1, drag=drag)


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

    global \
        latest_gesture, \
        latest_landmarks, \
        first_drag, \
        top_margin, \
        bottom_margin, \
        last_gesture_time, \
        cursor_x, \
        cursor_y, \
        inp
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
            (screen_height * (result.hand_landmarks[-1][8].y - top_margin))
            * 1.0
            / (bottom_margin - top_margin),
        )

        # in WEBOS the point_x and point_y are (0,0) at the middle,
        # thus we substract half the screen width and height to get the correct position
        point_x -= screen_width / 2
        point_y -= screen_height / 2

        now = datetime.datetime.now()
        if last_gesture_time is None:
            time_delta = 10
        else:
            time_delta = (now - last_gesture_time).total_seconds()
        last_gesture_time = now

        print("The point is at:", point_x, point_y)
        print("The cursor is at:", cursor_x, cursor_y)
        print(" The difference is", int(point_x - cursor_x), int(point_y - cursor_y))
        print()

        if time_delta < 5.5:
            move_arb_distance(int(point_x - cursor_x), int(point_y - cursor_y), drag=1)
        else:
            move_arb_distance(int(point_x), int(point_y), drag=1)
        cursor_x, cursor_y = point_x, point_y

        # TODO: no longer need to mark first_drag as mousedown in WebOS
        # however we do need to track the position of the cursor at all times
        # if first_drag is False:
        #     first_drag = True
        #     pyautogui.mouseDown(point_x, point_y)

        # here this move to should be dragging the entire time,
        # note that usually we need to end the drag by moving drag=0 some position
        # we can most likely do this whenever we are moving again which would be with closed fist
    if gestureName == "Closed_Fist":
        # want to click once
        if result.hand_landmarks[-1][8].y < top_margin:
            return
        if result.hand_landmarks[-1][8].y > bottom_margin:
            return
        point_x, point_y = (
            screen_width * result.hand_landmarks[-1][4].x,
            (screen_height * (result.hand_landmarks[-1][4].y - top_margin))
            * 1.0
            / (bottom_margin - top_margin),
        )
        point_x -= screen_width / 2
        point_y -= screen_height / 2
        now = datetime.datetime.now()
        time_delta = (now - last_gesture_time).total_seconds()
        last_gesture_time = now
        # TODO overload these function to account for the fact they can only move 15 dx and 15 dy
        if time_delta < 5.5:
            move_arb_distance(int(point_x - cursor_x), int(point_y - cursor_y), drag=0)
        else:
            # when its been more than 5.5 seconds since the last gesture the mouse disappears
            # and the position is reset to the center aka 0,0
            move_arb_distance(int(point_x), int(point_y), drag=0)
        inp.click()
        cursor_x, cursor_y = point_x, point_y


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

    global inp
    global client
    client = setup_webos()
    launch_info = launch_cut_the_rope(client)

    # now create a connection to mouse controls:
    # should only need one client
    inp = InputControl(client)
    inp.connect_input()

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

        cv.rectangle(
            display,
            (0, int(h * top_margin)),
            (w, int(h * bottom_margin)),
            (0, 0, 255),
            2,
        )
        cv.putText(
            display,
            latest_gesture,
            (20, int(h * top_margin + 30)),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv.imshow("Hand Gesture Recognition with MediaPipe", display)
        # use esc key to close application
        if cv.waitKey(10) == 27:
            break

    cap.release()
    inp.disconnect_input()


if __name__ == "__main__":
    main()
