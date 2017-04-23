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


# polynomial from prev frame
21-04-2017 21:02

Using the polynimal from the previous frame improved the performance immensely.
Now I only have 3 spots where the polygon goes a bit offlane on one edge,
and 1 spot (under the tree with heavy shadow) where it goes completely crazy.

I'm not using any kind of smoothing over the past N polynomials. This might be
one way to approach the problem. Looking into other thresholdings (different
space, Sobel, other that I may find).

I've also tried reducing the margin for getting points from the last polynomial.
This helps quite a bit, but is not enough for passing the hard interval since
it gets progressively worse. I get trash on the left top of the image and that
slowly pulls the left polynomial to the left. This might make the recovery
slower, but it seems to recover fine after the hard interval.

I need to work on a way for the pipeline to decide when to use the points from
the previous polynomial or the sliding windows.


# shadow hell
22-04-2017 21:31

I've noticed that when I use sliding windows for the shadow images, the result
is a lot worse than when I use the previous polynomials as a baseline. This is
because the polynomials restricts the number of points used in the threshold to
a more probable area than the sliding window.

# sobel x is cool
23-04-2017 10:00

I adjusted the parameters of the Sobel thresholding in the x axis and combined
it with the red thresholding. The result was quite good. I've noticed in some
frames that the margin around the previous polynomial is not wide enough to get
points that would help make a better fit.

I've also noticed that the red thresholding is a big puddle on the lighter road.
The saturation threshold seems to work better.

Try:
 - incrase margin around previous polynomial
 - make some decision when to use sat or red thresh


# project video done
23-04-2017 12:58

I increased the margin to 50, adjusted the parameters of the red thresh and
the binary input is now an OR of sobel x, red and sat thresh.

I noticed that sobel x can't pick up yellow when the road has a light color.
This becomes specially bad on the challenge video. Also the saturation goes
crazy when shadows are present.

I'd like to try doing something with the white and yellow colors.
