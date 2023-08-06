import cv2 as cv
from cv2 import aruco
import numpy as np

marker_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

# used this dictionary as it is mentioned in the competition rules that they will display a black and white Aruco Tag using the 4x4_50 tag library on every marker

param_markers = aruco.DetectorParameters_create()

# cap = cv.VideoCapture(0)  for live webcam detection
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

# Loading the sample Image
frame = cv.imread(r"C:\Users\Akshat\Downloads\arucoMarkerImage.png")

gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
marker_corners, marker_IDs, reject = aruco.detectMarkers(
    gray_frame, marker_dict, parameters=param_markers
)

# the detectMarkers return 3 values
# marker_corners: A list containing the corner coordinates of the detected markers.
# marker_IDs: A list containing the IDs of the detected markers.
# reject: A list of rejected markers.

if marker_corners:
    for ids, corners in zip(marker_IDs, marker_corners):
        # Drawing the outline of the detected marker
        cv.polylines(frame, [corners.astype(np.int32)], True, (0, 0, 255), 8, cv.LINE_AA)

        # Reshaping and extracting corner points
        corners = corners.reshape(4, 2)
        corners = corners.astype(int)
        top_right = corners[0].ravel()
        top_left = corners[1].ravel()
        bottom_right = corners[2].ravel()
        bottom_left = corners[3].ravel()

        # Drawing the ID of the marker at the top-right corner
        cv.putText(
            frame,
            f"id: {ids[0]}",
            top_right,
            cv.FONT_HERSHEY_COMPLEX,
            1.3,
            (200, 100, 0),
            5,
            cv.LINE_AA,
        )

        # Printing the marker ID and its corner points
        print(ids, "  ", corners)

# Create a resizable window and set it to full screen
cv.namedWindow("frame", cv.WINDOW_NORMAL)
cv.setWindowProperty("frame", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

# Display the image with detected markers
cv.imshow("frame", frame)
# cap.release() if live webcam was used

# Wait for a key press
key = cv.waitKey(0)
if key == ord("q"):
    cv.destroyAllWindows()
