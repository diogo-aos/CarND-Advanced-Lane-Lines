import pickle

def save_calibration_params(mtx, dist, fname):
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    with open(fname, "wb") as f:
        pickle.dump(dist_pickle, f)

def load_calibration_params(fname):
    'returns the mtx and dist calibration parameters'
    with open(fname, "rb") as f:
        dist_pickle = pickle.load(f)

    if 'mtx' not in dist_pickle:
        raise TypeError('mtx not in calibration file')
    if 'dist' not in dist_pickle:
        raise TypeError('dist not in calibration file')
    return dist_pickle['mtx'], dist_pickle['dist']
