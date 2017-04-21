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


def write_on_image(img, txt=[]):
    'write strings in txt in corresponding lines in lines'
    img = img.copy()
    x, y = 0, 35
    for l, t in enumerate(txt):
        cv2.putText(img, t, (0, y * (l + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    return img

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.

    Taken from project 1.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    else:
        raise ValueError('orient must be x or y')
    # 3) Take the absolute value of the derivative or gradient
    sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sobel = (sobel * 255 / sobel.max()).astype(np.uint8)
    # 5) Create a mask of 1's where the scaled gradient magnitude
            # is > thresh_min and < thresh_max
    binout = np.zeros_like(sobel)
    binout[(sobel >= thresh_min) & (sobel <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    return binout

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    ihls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # 2) Apply a threshold to the S channel
    bin_output = np.zeros(img.shape[:-1], dtype=np.uint8)
    bin_output[(ihls[...,2] > thresh[0]) & (ihls[...,2] <= thresh[-1])] = 1
    # 3) Return a binary image of threshold result
    return bin_output

def bgr_threshold(img, thresh=(0, 255), channel=2):
    'img is BGR, default channel is red'
    h, w = img.shape[:2]
    bin_img = np.zeros((h,w), dtype=np.uint8)
    bin_img[(img[...,channel] > thresh[0]) & (img[...,channel] < thresh[1])] = 255
    return bin_img

def apply_roi(img):
    h, w = img.shape[:2]
    #                   x                     y
    bot_h, top_h = h - int(crop_bot * h), int(crop_top * h)
    vertices = [(int(w * bot_left), bot_h),  # bottom left vertex
                (w - int(w * bot_right), bot_h),  # bottom right vertex
                (w - int(w * top_right), top_h),  # top right vertex
                (int(w * top_left), top_h)  # top left vertex
               ]
    vertices = np.array(vertices, dtype=np.int32)
    masked = region_of_interest(img, [vertices])
    return masked

def warp(img):
    h, w = img.shape[:2]
    warped = cv2.warpPerspective(img, warp_M, (w, h), flags=cv2.INTER_LINEAR)
    return warped

def unwarp(img):
    h, w = img.shape[:2]
    unwarped = cv2.warpPerspective(img, unwarp_M, (w, h), flags=cv2.INTER_LINEAR)
    return unwarped

def draw_trapezoid(img):
    img = img.copy()
    cv2.circle(img, (bl, bot), 5, 100, 5)
    cv2.circle(img, (tl, top), 5, 100, 5)
    cv2.circle(img, (br, bot), 5, 100, 5)
    cv2.circle(img, (tr, top), 5, 100, 5)
    return img

def get_points_for_fit(binary_warped, nwindows=9):
    h, w = binary_warped.shape
    histogram = np.sum(binary_warped, axis=0)

    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    midpoint = int(w / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_h = int(h / nwindows)
    nonzeroy, nonzerox = binary_warped.nonzero()

    leftx_current = leftx_base  # will be updated
    rightx_current = rightx_base  # will be updated

    margin = 100  # width of window for each side
    minpix = 50  # minimum of pixels found on window to recenter

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        # degine top and bottom window heights
        win_y_low = h - (window + 1) * window_h
        win_y_high = h - window * window_h
        # degine left and right boundaries for left lane window
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        # degine left and right boundaries for right lane window
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # if minimum pixels in window, recenter window around new mean
        if len(good_left_inds) >= minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) >= minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # concatenate the arrays of indices of all windows
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

def poly_from_points(leftx, lefty, rightx, righty, meters=False):
    if meters:
        leftx = np.float64(leftx) * xm_per_pix
        rightx = np.float64(rightx) *xm_per_pix
        lefty = np.float64(lefty) *ym_per_pix
        righty = np.float64(righty) *ym_per_pix
    # fit second layer polynomial
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit


def fit_poly(img, meters=False, *args, **kwargs):
    leftx, lefty, rightx, righty = get_points_for_fit(img, *args, **kwargs)
    left_fit, right_fit = poly_from_points(leftx, lefty, rightx, righty, meters=meters)

    return left_fit, right_fit

def compute_curvature(y, left_fit, right_fit, meters=False):
    y_eval = y

    if meters:
        y_eval *= ym_per_pix

    # Calculate the radii of curvature
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    return left_curverad, right_curverad

def get_poly_funcs(left_fit, right_fit):
    left_fn = lambda x: left_fit[0] * x**2 + left_fit[1] * x + left_fit[2]
    right_fn = lambda x: right_fit[0] * x**2 + right_fit[1] * x + right_fit[2]
    return left_fn, right_fn

def draw_poly_lines(img, left_fit, right_fit, fill=True):
    img = img.copy()
    h, w = img.shape[:2]
    ploty = np.linspace(0, h-1, h)
    leftx, rightx = left_fit(ploty), right_fit(ploty)

    left_pts = [p for p in zip(leftx, ploty)]
    right_pts = [p for p in zip(rightx, ploty)]
    pts = left_pts
    pts.extend(right_pts[::-1])

    if fill:
        cv2.fillPoly(img,  np.int32([pts]), (255, 0, 0))
    else:
        cv2.polylines(img,  np.int32([pts]), True, (255, 0, 0), 2)

    return img

def warp_point(x, y, M):
    # http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html?highlight=warpperspective#warpperspective
    dx = (M[0,0] * x + M[0, 1] * y + M[0,2]) /
         (M[2,0] * x + M[2, 1] * y + M[2,2])
    dy = (M[1,0] * x + M[1, 1] * y + M[1,2]) /
         (M[2,0] * x + M[2, 1] * y + M[2,2])
    return dx, dy

def compute_center_offset(left_fit, right_fit, meters=False):
    canvas = np.zeros((h, w), dtype=np.uint8)
    y_eval = h - 1

    camera_center = w / 2

    # get x coord of left and right lanes on the polynomial at y_eval
    lx = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
    rx = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]

    # get unwarped coordenates of left and right points
    leftx, lefty = warp_point(lx, y_eval, unwarp_M)
    rightx, righty = warp_point(rx, y_eval, unwarp_M)

    # get unwarped coords of left and right points through drawing in an
    # empty image, unwarping that image and then checking nonzero pixels
#     cv2.circle(canvas, (int(lx), h-1), 1, 255, 1)
#     cv2.circle(canvas, (int(rx), h-1), 1, 255, 1)
#     canvas_unwarped = unwarp(canvas)

#     nz_x = canvas_unwarped.nonzero()[1]

#     leftx = nz_x[nz_x < camera_center][0]
#     rightx = nz_x[nz_x > camera_center][0]

    aprox_fitted_center = leftx + (rightx - leftx) / 2

    offset = aprox_fitted_center - camera_center
    if meters:
        offset *= xm_per_pix

    return offset
