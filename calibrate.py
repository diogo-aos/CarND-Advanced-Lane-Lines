import cv2
import sys
import glob
import numpy as np

from utils import *

if len(sys.argv) < 2:
    print('usage: calibrate.py [/path/to/calibrate/image.jpg]')
    sys.exit(0)

arg_paths = sys.argv[1:]
paths = []
for p in arg_paths:
    paths.extend(glob.glob(p))

imgs = [cv2.imread(p) for p in paths]

print('got {} paths'.format(len(paths)))

if None in imgs:
    print('could not read {} images'.format(imgs.count(None)))


board_shape = (9, 6)

# create object coordinates for chessboard
objp = np.zeros((np.prod(board_shape), 3), np.float32)
objp[:,:2] = np.mgrid[0:board_shape[0], 0:board_shape[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

window = cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

print('finding corners in all images...')
# Step through the list and search for chessboard corners
for idx, img in enumerate(imgs):
    print('image {}/{}'.format(idx, len(imgs)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, board_shape, None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (8,6), corners, ret)
        #write_name = 'corners_found'+str(idx)+'.jpg'
        #cv2.imwrite(write_name, img)
        cv2.imshow('frame', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

import pickle

# Test undistortion on an image
some_img = imgs[0]
h, w, c = some_img.shape
img_size = (w, h)

print('computing calibration parameters...')
# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

some_undistorted = cv2.undistort(some_img, mtx, dist, None, mtx)

mosaic = np.empty((h*2, w, c), np.uint8)
mosaic[:h,...] = some_img
mosaic[h:,...] = some_undistorted

cv2.imshow('frame2', mosaic)
cv2.waitKey(2000)
cv2.destroyAllWindows()

cv2.imwrite('calibration_mosaic.jpg', mosaic)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
fname = "calibration_params.p"
save_calibration_params(mtx, dist, fname)
print('saved calibration parameters in {}'.format(fname))
