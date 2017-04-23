import cv2
import numpy as np
from collections import OrderedDict

def pipeline(img, left_fit=None, right_fit=None, nwindows=9):
    img = img.copy()
    h, w = img.shape[:2]
    txt = []
    ret_frames = OrderedDict()

    ret_frames['input'] = img

    ## undistort input image
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    ret_frames['undistorted'] = undistorted

    ## draw warp polygon on undistorted
    und_poly = undistorted.copy()
    cv2.polylines(und_poly,  np.int32([src_coords]), True, (255, 0, 0), 2)
    ret_frames['undistorted polygon warp'] = und_poly

    ## warp undistorted image
    undistorted_warped = warp(undistorted)
    ret_frames['undistorted_warped'] = undistorted_warped

    ## apply Sobel x abs thresholding to warped undistorted
    sobel_x = abs_sobel_thresh(undistorted_warped, orient='x',
                               thresh_min=sobel_abs_x['tbot'],
                               thresh_max=sobel_abs_x['ttop'])
    ret_frames['sobel_x'] = sobel_x

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


    ret_frames['poly input'] = red_bin | sobel_x | sat_bin

    ## sliding window, fit polynomial, radius, center offset, polygon mask
    img = ret_frames['poly input']
    # get left and right lane points for polynomial fit
    # lx, ly, rx, ry = get_points_for_fit(img, left_fit=left_fit,
    #                                     right_fit=right_fit,
    #                                     nwindows=nwindows)

    points_prev = None
    if left_fit is not None and right_fit is not None:
        points_prev = get_points_for_fit_from_polynomials(img, left_fit,
                                                          right_fit,
                                                          margin=20)
    lx, ly, rx, ry = get_points_for_fit_sliding_window(img, nwindows=nwindows)

    points_drawn = np.zeros((h,w,3), dtype=np.uint8)
    points_drawn[ly,lx,0] = 255
    points_drawn[ry,rx,1] = 255

    ret_frames['points_windows'] = points_drawn

    if points_prev is not None:
        lx, ly, rx, ry = points_prev
        points_drawn = np.zeros((h,w,3), dtype=np.uint8)
        points_drawn[ly,lx,0] = 255
        points_drawn[ry,rx,1] = 255

        ret_frames['points_prev'] = draw_poly_lines(points_drawn, left_fit, right_fit, fill=False)

    if len(lx) == 0 or len(rx) == 0:
        ret['final'] = ret_frames['poly input']
        return ret

    lp, rp = poly_from_points(lx, ly, rx, ry)  # left and right polynomials in pixel space
    lm, rm = fit_poly(img, meters=False)  # left and right polynomials in meter space

    ret_frames['pixel space polynomials'] = (lp, rp)

    # compute real world radius of curvature
    l_radius, r_radius = compute_curvature(y=img.shape[0]-1,
                                           left_fit=lm, right_fit=rm,
                                           meters=False)
    top_radii = compute_curvature(y=int(img.shape[0] * 0.2),
                                   left_fit=lm, right_fit=rm,
                                   meters=False)
    # compute center offset
    center_offset = compute_center_offset(lp, rp, meters=True)

    # create a mask with a filled polygon drawn from the polynomials

    polygon_mask_0 = np.zeros((h, w), dtype=np.uint8)
    #polygon_mask[img>0] = 255
    l_fn, r_fn = get_poly_funcs(lp, rp)
    polygon_mask_0 = draw_poly_lines(polygon_mask_0, l_fn, r_fn, fill=True)
    polygon_mask = points_drawn.copy()
    polygon_mask[polygon_mask_0>0, 2] = 255

    txt.append('radii left={:.1f} right={:.1f}'.format(l_radius, r_radius))
    txt.append('offset {:.2f}m'.format(center_offset))

    top_l_r, top_r_r = top_radii
    txt.append('')
    txt.append('')
    txt.append('top radii l={:.1f} r={:.1f}'.format(top_l_r, top_r_r))

    ret_frames['polygon_mask'] = polygon_mask

    ## unwarp polygon mask
    unwarped_polygon_mask = unwarp(polygon_mask)
    ret_frames['unwarped_polygon_mask'] = unwarped_polygon_mask

    ## overlay polygon mask with undistorted image
    overlay = undistorted | unwarped_polygon_mask
    #overlay[unwarped_polygon_mask > 0] = 255

    ret_frames['overlay'] = overlay

    final_image = overlay
    out_img = write_on_image(final_image, txt)
    ret_frames['final'] = out_img

    return ret_frames
