import cv2 as cv
from cv2 import aruco
import numpy as np

marker_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

param_markers = aruco.DetectorParameters_create()

# loading the sample image
frame = cv.imread(r"C:\Users\Akshat\Downloads\arucoMarkerImage.png")
# resizing the image
frame = cv.resize(frame, (720, 720))

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
        cv.polylines(
            frame, [corners.astype(np.int32)], True, (0, 0, 255), 10, cv.LINE_AA
        )
        corners = corners.reshape(4, 2)
        corners = corners.astype(int)
        top_right = corners[0].ravel()
        top_left = corners[1].ravel()
        bottom_right = corners[2].ravel()
        bottom_left = corners[3].ravel()
        cv.putText(
            frame,
            f"id: {ids[0]}",
            top_right,
            cv.FONT_HERSHEY_COMPLEX,
            1.3,
            (200, 100, 0),
            3,
            cv.LINE_AA,
        )
        print(ids, "  ", corners)

cv.imshow("frame", frame)
key = cv.waitKey(0)
if key == ord("q"):
    cv.destroyAllWindows()
