import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# define ARUCO params
SQUARE_LENGTH = 500
MARKER_LENGTH = 300
NUMBER_OF_SQUARES_VERTICALLY = 11
NUMBER_OF_SQUARES_HORIZONTALLY = 8

calibrated = False

# ========== CODE TO GENERATE CHARUCO BOARD IMAGE ============
charuco_marker_dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
charuco_board = cv.aruco.CharucoBoard(
    size=(NUMBER_OF_SQUARES_HORIZONTALLY, NUMBER_OF_SQUARES_VERTICALLY),
    squareLength=SQUARE_LENGTH,
    markerLength=MARKER_LENGTH,
    dictionary=charuco_marker_dictionary,
)

image_name = f"ChArUco_Marker_{NUMBER_OF_SQUARES_HORIZONTALLY}x{NUMBER_OF_SQUARES_VERTICALLY}.png"
charuco_board_image = charuco_board.generateImage(
    [
        i * SQUARE_LENGTH
        for i in (NUMBER_OF_SQUARES_HORIZONTALLY, NUMBER_OF_SQUARES_VERTICALLY)
    ]
)
cv.imwrite(image_name, charuco_board_image)

# ========== CODE TO CALIBRATE CAMERA ============

def calibrate(
    cap, charuco_board, charuco_detector, total_image_points, total_object_points
):
    _, k, dist, rvec, tvec = cv.calibrateCamera(
        total_object_points,
        total_image_points,
        frame_gray.shape,
        None,
        None,
    )
    global calibrated
    calibrated = True
    return k, dist


def draw_plane(frame, k, dist, rvec, tvec):
    h, w = frame.shape[:2]

    # set this plane size to the same size of our physical legal playing board
    board_w = SQUARE_LENGTH * NUMBER_OF_SQUARES_HORIZONTALLY + 1000 
    board_h = SQUARE_LENGTH * NUMBER_OF_SQUARES_VERTICALLY + 1000

    plane_3d = np.array(
        [[0, 0, 0], [board_w, 0, 0], [board_w, board_h, 0], [0, board_h, 0]],
        dtype=np.float32,
    )

    img_pts, _ = cv.projectPoints(plane_3d, rvec, tvec, k, dist)
    img_pts = np.int32(img_pts.reshape(-1, 2))
    overlay = frame.copy()
    contour = np.round(img_pts).astype(np.int32).reshape(-1, 1, 2)
    cv.fillConvexPoly(overlay, contour, (0, 255, 0))
    cv.polylines(overlay, [contour], True, (0, 180, 0), 2)
    # frame = cv.addWeighted(frame, 0.7, overlay, 0.3, 0)

    return cv.addWeighted(overlay, 0.35, frame, 0.7, 0)


total_image_points = []
total_object_points = []

# using continuity camera with my phone
cap = cv.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

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
            k, dist = calibrate(
                cap,
                charuco_board,
                charuco_detector,
                total_image_points,
                total_object_points,
            )

        print("object_points shape:", object_points.shape)
        print("image_points shape:", image_points.shape)
        if calibrated:
            if (
                object_points is not None
                and image_points is not None
                and len(object_points) >= 6
            ):
                # pose estimation
                ok, rvec, tvec = cv.solvePnP(object_points, image_points, k, dist)
                print("rvec:", rvec, " tvec: ", tvec)
                if ok:
                    # cv.drawFrameAxes(frame, k, dist, rvec, tvec, SQUARE_LENGTH)
                    frame = draw_plane(frame, k, dist, rvec, tvec)

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

    cv.imshow("Calibration", frame)
    if cv.waitKey(10) == 27:
        break

cap.release()
cv.destroyAllWindows()
