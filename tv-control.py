# %%
import datetime
import json
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, "./PyWebOSTV")

import time

import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
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


# use this to create multiple boards, maybe also can export them programmatically
# later so they can be auto loaded on our ipads for instant camera calibration
def create_charuco_board():
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
    charuco_board = cv.aruco.CharucoBoard(
        size=(NUMBER_OF_SQUARES_HORIZONTALLY, NUMBER_OF_SQUARES_VERTICALLY),
        squareLength=SQUARE_LENGTH,
        markerLength=MARKER_LENGTH,
        dictionary=dictionary,
    )
    image_name = f"ChArUco_Marker_{NUMBER_OF_SQUARES_HORIZONTALLY}x{NUMBER_OF_SQUARES_VERTICALLY}.png"
    img = cv.aruco.CharucoBoard.generateImage(
        charuco_board,
        [
            i * SQUARE_LENGTH
            for i in (NUMBER_OF_SQUARES_HORIZONTALLY, NUMBER_OF_SQUARES_VERTICALLY)
        ],
    )
    cv.imwrite(image_name, img)
    return charuco_board


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


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# screenwidth and height cannot be fetched by session
# input custom dimensions here
# FOR TV
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

total_image_points = []
total_object_points = []

# define ARUCO params
SQUARE_LENGTH = 500
MARKER_LENGTH = 300
NUMBER_OF_SQUARES_VERTICALLY = 11
NUMBER_OF_SQUARES_HORIZONTALLY = 8
calibrated = False
last_gesture = None
# physical board
play_board_w = None  # physical
play_board_h = None
board_corners = []
cam_k = None
cam_dist = None


# get monitor specs to use and setup hand tracking software also
def setup():
    # using continuity camera with my phone
    cap = cv.VideoCapture(1)
    model_path = "hand_landmarker_task.task"
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.LIVE_STREAM,
        result_callback=result_callback,
    )

    recognizer = HandLandmarker.create_from_options(options)

    # setup charuco board and detector
    charuco_board = create_charuco_board()
    charuco_detector = cv.aruco.CharucoDetector(charuco_board)

    return cap, recognizer, charuco_board, charuco_detector


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
    result: mp.tasks.vision.HandLandmarkerResult,
    output_image: mp.Image,
    timestamp_ms: int,
):
    global \
        latest_landmarks, \
        first_drag, \
        board_width, \
        board_height, \
        last_gesture_time, \
        cursor_x, \
        cursor_y, \
        inp
    if not result.hand_landmarks:
        latest_landmarks = None
        return

    x, y = check_points_and_project()

    point_x, point_y = (
        screen_width * x,
        screen_height * y,
    )
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

    # TODO: differentiate between drag and click via time delta
    if time_delta >= 10:
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

    # if gestureName == "Pointing_Up":
    #     # x and y derives from landmark position
    #     # duration should be 0 so its somewhat immediate
    #     # 1/(7/8 - 1/10)

    #     if result.hand_landmarks[-1][8].y < top_margin:
    #         return
    #     if result.hand_landmarks[-1][8].y > bottom_margin:
    #         return

    #     point_x, point_y = (
    #         screen_width * result.hand_landmarks[-1][8].x,
    #         (screen_height * (result.hand_landmarks[-1][8].y - top_margin))
    #         * 1.0
    #         / (bottom_margin - top_margin),
    #     )

    #     # in WEBOS the point_x and point_y are (0,0) at the middle,
    #     # thus we substract half the screen width and height to get the correct position
    #     point_x -= screen_width / 2
    #     point_y -= screen_height / 2

    #     now = datetime.datetime.now()
    #     if last_gesture_time is None:
    #         time_delta = 10
    #     else:
    #         time_delta = (now - last_gesture_time).total_seconds()
    #     last_gesture_time = now

    #     print("The point is at:", point_x, point_y)
    #     print("The cursor is at:", cursor_x, cursor_y)
    #     print(" The difference is", int(point_x - cursor_x), int(point_y - cursor_y))
    #     print()

    #     if time_delta < 5.5:
    #         move_arb_distance(int(point_x - cursor_x), int(point_y - cursor_y), drag=1)
    #     else:
    #         move_arb_distance(int(point_x), int(point_y), drag=1)
    #     cursor_x, cursor_y = point_x, point_y

    # TODO: no longer need to mark first_drag as mousedown in WebOS
    # however we do need to track the position of the cursor at all times
    # if first_drag is False:
    #     first_drag = True
    #     pyautogui.mouseDown(point_x, point_y)

    # here this move to should be dragging the entire time,
    # note that usually we need to end the drag by moving drag=0 some position
    # we can most likely do this whenever we are moving again which would be with closed fist
    # if gestureName == "Closed_Fist":
    #     # want to click once
    #     if result.hand_landmarks[-1][8].y < top_margin:
    #         return
    #     if result.hand_landmarks[-1][8].y > bottom_margin:
    #         return
    #     point_x, point_y = (
    #         screen_width * result.hand_landmarks[-1][4].x,
    #         (screen_height * (result.hand_landmarks[-1][4].y - top_margin))
    #         * 1.0
    #         / (bottom_margin - top_margin),
    #     )
    #     point_x -= screen_width / 2
    #     point_y -= screen_height / 2
    #     now = datetime.datetime.now()
    #     time_delta = (now - last_gesture_time).total_seconds()
    #     last_gesture_time = now
    #     # TODO overload these function to account for the fact they can only move 15 dx and 15 dy
    #     if time_delta < 5.5:
    #         move_arb_distance(int(point_x - cursor_x), int(point_y - cursor_y), drag=0)
    #     else:
    #         # when its been more than 5.5 seconds since the last gesture the mouse disappears
    #         # and the position is reset to the center aka 0,0
    #         move_arb_distance(int(point_x), int(point_y), drag=0)
    #     inp.click()
    #     cursor_x, cursor_y = point_x, point_y


# we can use the following gesture 👆 from
# https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/python


# ========== CODE TO CALIBRATE CAMERA ============
def calibrate(frame_gray):
    global calibrated, cam_k, cam_dist
    _, cam_k, cam_dist, rvec, tvec = cv.calibrateCamera(
        total_object_points,
        total_image_points,
        frame_gray.shape,
        None,
        None,
    )
    calibrated = True


def get_playing_board_boundaries(frame, board, charuco_detector):
    global play_board_w, play_board_h, board_corners
    marker_corners, marker_ids, _ = charuco_detector.detectMarkers(frame)
    if marker_ids is None:
        return None

    board_detector = cv.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _ = board_detector.detectBoard(frame)
    if charuco_ids is not None and len(charuco_ids) > 4:
        obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
        ok, rvec, tvec = cv.solvePnP(obj_pts, img_pts, cam_k, cam_dist)
        if not ok:
            return None

    R, _ = cv.Rodrigues(rvec)

    play_board_w = SQUARE_LENGTH * NUMBER_OF_SQUARES_HORIZONTALLY * 3
    play_board_h = SQUARE_LENGTH * NUMBER_OF_SQUARES_VERTICALLY * 2

    # play board cords
    corners_board = np.array(
        [
            [0, 0, 0],
            [play_board_w, 0, 0],
            [play_board_w, play_board_h, 0],
            [0, play_board_h, 0],
        ]
    )

    # apply transformation to camera coordinates
    board_corners = (R @ corners_board.T + tvec).T


# TODO: need to connect arkit functionality in swift for lidar to python function
def check_points_and_project():
    # record camera 3D x, y for last detected hand_landmark for the pointer finger tip
    # get x, y, z data relative to the camera via lidar from arkit
    # TODO: Explore if it's better to also define plane and arkit position data for the playing board

    # create plane from board corners
    # get playing board boundaries
    [top_left, top_right, bottom_right, bottom_left] = board_corners

    v1 = top_right - top_left
    v2 = bottom_left - top_left

    x, y, z = 1, 1, 1

    # cross product for normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # calculate the dot product using 3rd point
    d = np.dot(cp, top_right)

    print("Planar equaiton {0}x + {1}y + {2}z = {3}".format(a, b, c, d))

    # check if point intersects plane
    point = [a * x, b * y, c * z]
    if d == point:
        print("Point is on the plane")

    # check if point is within board boundaries
    if top_left[0] <= x <= bottom_right[0] and top_left[1] <= y <= bottom_right[1]:
        print("Point is within board boundaries")

    # map 3d landmark points back to 2d screen coords
    # can use arkit projectPoint()
    # convert to tv screen coords

    return x, y


def detect_charuco(frame, charuco_board):
    # get charuco info
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    charuco_detector = cv.aruco.CharucoDetector(charuco_board)
    charuco_corners, charuco_ids, aruco_corners, aruco_ids = (
        charuco_detector.detectBoard(frame_gray)
    )

    if charuco_ids is not None and charuco_corners is not None:
        object_points, image_points = charuco_board.matchImagePoints(
            charuco_corners, charuco_ids
        )
        total_object_points.append(object_points)
        total_image_points.append(image_points)

        if object_points.shape[0] >= 4 and not calibrated:
            print("calling calibrate function here")
            calibrate(frame_gray)
            get_playing_board_boundaries(frame_gray, charuco_board, charuco_detector)

        print("object_points shape:", object_points.shape)
        print("image_points shape:", image_points.shape)

    if not calibrated:
        cv.putText(
            frame,
            "Collecting Views for Camera Calibration",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
    else:
        cv.putText(
            frame,
            "Calibration Success!",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )


# control mouse movements
def main():
    cap, recognizer, boards, charuco_detector = setup()

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
        # frame = cv.flip(frame, 1)
        # resize the image
        frame = cv.resize(frame, (1280, 720))

        # run charuco detection

        detect_charuco(frame, boards)

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

        cv.imshow("Connect the Rope TV", display)
        # use esc key to close application
        if cv.waitKey(10) == 27:
            break

    cap.release()
    inp.disconnect_input()


if __name__ == "__main__":
    main()
