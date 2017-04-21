import cv2
import numpy as np
from collections import OrderedDict

def pipeline(img):
    img = img.copy()
    h, w = img.shape[:2]
    txt = []
    ret_frames = OrderedDict()


    ret_frames['input'] = img

    ## undistort input image
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    ret_frames['undistorted'] = undistorted

    ## warp undistorted image
    undistorted_warped = warp(undistorted)
    ret_frames['undistorted_warped'] = undistorted_warped

    ## apply Sobel direction thresholding to warped undistorted
    sobel_dir_threshold = dir_threshold(undistorted_warped,
                                        sobel_kernel=sobel_dir['sobel_kernel'],
                                        thresh=(sobel_dir['tbot'], sobel_dir['ttop']))
    ret_frames['sobel_dir'] = sobel_dir_threshold

    ## apply HSL saturation thresholding to warped undistorted
    sat_bin = hls_select(undistorted_warped,
                         thresh=(saturation['tbot'], saturation['ttop']))
    ret_frames['sat_bin'] = sat_bin

    ## apply red thresholding to warped undistorted
    red_bin = bgr_threshold(undistorted_warped,
                            thresh=(red_threshold['tbot'], red_threshold['ttop']),
                            channel=2)
    ret_frames['red_bin'] = red_bin

    red_sat_bin = np.empty_like(red_bin)
    red_sat_bin[(red_bin > 0) | (sat_bin > 0)] = 255
    ret_frames['poly input'] = red_sat_bin

    ## sliding window, fit polynomial, radius, center offset, polygon mask
    img = ret_frames['poly input']
    # get left and right lane points for polynomial fit
    lx, ly, rx, ry = get_points_for_fit(img)  # sliding window

    points_drawn = np.zeros((h,w,3), dtype=np.uint8)
    points_drawn[ly,lx,0] = 255
    points_drawn[ry,rx,1] = 255


    lp, rp = poly_from_points(lx, ly, rx, ry)  # left and right polynomials in pixel space
    lm, rm = fit_poly(img, meters=False)  # left and right polynomials in meter space

    # compute real world radius of curvature
    l_radius, r_radius = compute_curvature(y=img.shape[0]-1,
                                           left_fit=lm, right_fit=rm,
                                           meters=False)
    # compute center offset
    center_offset = compute_center_offset(lp, rp, meters=True)

    # create a mask with a filled polygon drawn from the polynomials

    polygon_mask = np.zeros((h, w, 3), dtype=np.uint8)
    #polygon_mask[img>0] = 255
    l_fn, r_fn = get_poly_funcs(lp, rp)
    polygon_mask = draw_poly_lines(polygon_mask, l_fn, r_fn, fill=True)

    txt.append('radii left={:.1f} right={:.1f}'.format(l_radius, r_radius))
    txt.append('offset {:.2f}m'.format(center_offset))

    ret_frames['polygon_mask'] = polygon_mask

    ## unwarp polygon mask
    unwarped_polygon_mask = unwarp(polygon_mask)
    ret_frames['unwarped_polygon_mask'] = unwarped_polygon_mask

    ## overlay polygon mask with undistorted image
    overlay = undistorted.copy()
    overlay[unwarped_polygon_mask > 0] = 255

    ret_frames['overlay'] = overlay

    final_image = overlay
    out_img = write_on_image(final_image, txt)
    ret_frames['final'] = out_img

    return ret_frames
