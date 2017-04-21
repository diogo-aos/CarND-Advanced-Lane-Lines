import cv2
import numpy as np
import pipeline
import glob
import sys

helptxt = '''
python analyze.py [args]
    -i /path/1 /path/1/*.jpg
'''

if '-i' not in sys.argv:
    print(helptxt)
    sys.exit(0)

i_idx = sys.argv.index('-i')
arg_paths = sys.argv[i_idx + 1:]
paths = []
for p in arg_paths:
    paths.extend(glob.glob(p))

# load test images
images = [cv2.imread(p) for p in paths]
if None in images:
    print('images not read: ', images.count(None))


def write_on_image(img, txt=[], lines=[]):
    'write strings in txt in corresponding lines in lines'
    img = img.copy()
    x, y = 0, 20
    for l, t in zip(lines, txt):
        cv2.putText(img, t, (0, y * l), cv2.FONT_HERSHEY_PLAIN, 1, 100, 2)
    return img

window = cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
for img in images:
    frames = pipeline.pipeline(img, kernel_size=3, thresh(0, 255))

    cv2.imshow('frame', img)
    
    im = write_on_image(frames['undistorted'], ['undistorted'], [1])
    cv2.imshow('frame', im)
    key = cv2.waitKey()
    if key == ord('n'):
        continue
    elif key == ord('q'):
        cv2.destroyAllWindows()
        break
        

    
