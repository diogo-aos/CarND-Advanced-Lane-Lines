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


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the gradient in x and y separately
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    s = np.sqrt(sx**2 + sy**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    s = (s * 255 / s.max()).astype(np.uint8)
    # 5) Create a binary mask where mag thresholds are met
    binout = np.zeros_like(s)
    binout[(s >= mag_thresh[0]) & (s <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binout

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the gradient in x and y separately
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    sx = np.absolute(sx)
    sy = np.absolute(sy)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    s = np.arctan2(sy, sx)
    # 5) Create a binary mask where direction thresholds are met
    binout = np.zeros_like(s)
    binout[(s >= thresh[0]) & (s <= thresh[-1])] = 1
    # 6) Return this mask as your binary_output image
    return binout


def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    ihls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # 2) Apply a threshold to the S channel
    bin_output = np.zeros(img.shape[:-1], dtype=np.uint8)
    bin_output[(ihls[...,2] > thresh[0]) & (ihls[...,2] <= thresh[-1])] = 1
    # 3) Return a binary image of threshold result
    return bin_output
