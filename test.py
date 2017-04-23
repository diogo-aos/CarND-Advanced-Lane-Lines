import cv2
import numpy as np
import pickle
import utils
import glob
import os.path
import sys
import tqdm

# load configuration
with open('threshold_configs.p', 'rb') as f:
    c = pickle.load(f)

h, w = 720, 1280

# load thresholding configuration values
sobel_abs_x = c['sobel x']
sobel_abs_y = c['sobel y']
sobel_dir = c['sobel_dir']
saturation = c['hsl saturation']
mag_bin = c['mag_bin']
red_threshold = c['red']

# load warping parameters
src_coords = c['warp src']
dst_coords = c['warp dst']

warp_M = cv2.getPerspectiveTransform(src_coords, dst_coords)
unwarp_M = cv2.getPerspectiveTransform(dst_coords, src_coords)

# load real world space convertion values
xm_per_pix, ym_per_pix = c['m per px']

# load precomputed camera calibration
mtx, dist = utils.load_calibration_params('calibration_params.p')

# "import" functions to namespace
with open('funcs.py', 'r') as f:
    exec(f.read())

# "import" pipeline to namespace
with open('pipeline.py', 'r') as f:
    exec(f.read())

# load project video metadata
vid_files = glob.glob('video_imgs/project_video/*')
numbered_vid_fn = [(int(fn.split('/')[-1].rstrip('.jpg')), fn) for fn in vid_files]
numbered_vid_fn.sort()

arg = sys.argv[1]
if arg == 'viz':
    timeout = 42
    inc = 1
    i = 0

    idx, fn = numbered_vid_fn[i]
    img = cv2.imread(fn)
    ret = pipeline(img)

    while True:
        idx, fn = numbered_vid_fn[i]
        img = cv2.imread(fn)
        if img is None:
            print('image not read with index ', i)
            contnue
        left_fit, right_fit = ret.get('pixel space polynomials', (None, None))
        ret = pipeline(img, left_fit=left_fit, right_fit=right_fit)
        show_img = write_on_image(ret['final'], txt=['','','image {}/{}'.format(idx, len(numbered_vid_fn))])
        cv2.imshow('frame', show_img)
        key = cv2.waitKey(timeout)
        if key == ord('q'):
            break
        if key == ord('w'):
            # show all frames from pipeline
            items = [(desc, im) for desc, im in ret.items()]
            items = [it for it in items if isinstance(it[1], np.ndarray) and it[1].shape[:2] == (720, 1280)]
            print([it[0] for it in items])
            #for desc, im in ret.items():
            j = 0
            while True:
                desc, im = items[j]
                tmp = write_on_image(im, txt=[desc])
                cv2.imshow('frame', tmp)
                wkey = cv2.waitKey()
                if wkey == ord('q'):
                    break
                elif wkey == ord('m'):
                    j = min(j+1, len(items)-1)
                elif wkey == ord('n'):
                    j = max(j-1, 0)
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
        elif key == ord('k'):
            timeout = 0
            i += inc*10
            i = min(i, len(numbered_vid_fn) - 1)
        elif key == ord('j'):
            timeout = 0
            i -= inc*10
            i = max(0, i)
        else:
            i += inc
            i = min(i, len(numbered_vid_fn) - 1)
    cv2.destroyAllWindows()

elif arg == 'store':
    ret = {}
    # fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
    # video = cv2.VideoWriter('output_video.mpeg', fourcc, 24, (w, h))

    fourcc = cv2.VideoWriter_fourcc('X','2','6','4')
    video = cv2.VideoWriter('output_video.mp4', fourcc, 24, (w, h))

    for idx, fn in tqdm.tqdm(numbered_vid_fn):
        img = cv2.imread(fn)
        left_fit, right_fit = ret.get('pixel space polynomials', (None, None))
        ret = pipeline(img, left_fit=left_fit, right_fit=right_fit)
        out_fn = os.path.join('output_images', '{}.p'.format(idx))
        out_fn_im = os.path.join('output_images', '{}.jpg'.format(idx))
        out_img = write_on_image(ret['final'],
                txt=['','','image {}/{}'.format(idx, len(numbered_vid_fn))])
        with open(out_fn, 'wb') as f:
            pickle.dump(ret, f)
        cv2.imwrite(out_fn_im, out_img)
        video.write(out_img)
    video.release()

elif arg == 'multi':
    from multiprocessing import Pool
    import time
    def process(args):
        start, size = args
        ret = {}
        for idx, fn in numbered_vid_fn[start:start+size]:
            img = cv2.imread(fn)
            left_fit, right_fit = ret.get('pixel space polynomials', (None, None))
            ret = pipeline(img, left_fit=left_fit, right_fit=right_fit)
            out_fn = os.path.join('output_images', '{}.p'.format(idx))
            out_fn_im = os.path.join('output_images', '{}.jpg'.format(idx))
            out_img = write_on_image(ret['final'],
                    txt=['','','image {}/{}'.format(idx, len(numbered_vid_fn))])
            with open(out_fn, 'wb') as f:
                pickle.dump(ret, f)
            cv2.imwrite(out_fn_im, out_img)
        return idx
    nprocs = 4
    p = Pool(nprocs)
    size = int(len(numbered_vid_fn) / nprocs)
    start = [i * size for i in range(nprocs)]
    size = [size] * nprocs
    args = [(st, si) for st, si in zip(start, size)]
    print(args)
    time_start = time.time()
    print(p.map(process, args))
    total_time = time.time() - time_start
    print('total time: ', total_time)
    print('time per it: ', total_time / len(numbered_vid_fn))
