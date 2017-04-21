import pickle
import cv2
import os.path


with open(sys.argv[1], 'rb') as f:
    meta = pickle.load(f)

n_img = meta['length']
viz_folder = meta['folder']
timeout = 42
inc = 1
i = 0

while True:
    with open(os.path.join(viz_folder, '{}.p'.format(i)), 'rb') as f:
        ret = pickle.load(f)
    cv2.imshow('frame', ret['final'])
    key = cv2.waitKey(timeout)
    if key == ord('q'):
        break
    if key == ord('w'):
        # show all frames from pipeline
        for desc, im in ret.items():
            cv2.imshow('frame', tmp)
            cv2.waitKey()
    elif key == ord(' '):
        if timeout == 0:
            timeout = 42
        else:
            timeout = 0
    elif key == ord('m'):
        timeout = 0
        i += inc
        i = min(i, len(numbered_vid_fn) - 1)
    elif key == ord('n'):
        timeout = 0
        i -= inc
        i = max(0, i)
    else:
        i += inc
        i = min(i, len(numbered_vid_fn) - 1)
cv2.destroyAllWindows()