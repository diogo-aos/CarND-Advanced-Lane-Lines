# just starting
15-04-2017 14:08

I've started the project by exploring some of the transformation done in class.
Namely, I've been tring the Sobel direction, magnitude direction and saturation
threshold. Of these, the saturation threshold is clearly better than the rest.
In all test images, it gets the lines quite well. However, it seems to be having
problems with shadows.

I've been using some code to adjust the parameters of these transformations
interactevely and that helped a lot. I can focus on one image at a time and
explore much faster the effect of increasing or decreasing each parameter.

I will move on to cropping the image, since I want to focus on the region of 
interest.

# problems ahead
16-04-2017 17:37

I've implemented region of interest, warping and finding a polynomial for each
lane (actually, this was just copy pasting code for the most part). It took some
time to fiddle with the warping zone.

The warping zone is not quite there yet. On test image 5, a straight road, the 
warped result show a biconcave shape ')(' for the lane lines. Naturally, this
makes the interpolated polynomials have that shape as well.

TODO:
 - unwarp polynomials to original unwarped image
 - fiddle with the warping zone
 - experiment warping the image before thresholding it
 - think about coming up with a scheme of choosing one of the polynomials (the
one with more points around the histogram peak for example) and use that for both
lanes with the adequate offset


# defining pipeline
20-04-2017 08:50

I've unwarped the polynomials to the original unwarped image. I've saved the
frames from all videos to a folder so it's easy to navigate all images of the
video interactively and to pinpoint the frame that does not work.

In the interactive script, some frames simply don't show any drawn polygon,
but out of the scrit the same images look fine. I've noted down the frames
where the polynomials are bad.

I'm already doing an OR mask between the saturation and red thresholds, and
this incrased the performance.

Next I need to continue debugging why I don't get a polygon on some frames.
