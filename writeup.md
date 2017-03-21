##Writeup Template

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/slide_windows.jpg
[image2]: ./output_images/hot_windows.jpg
[image3]: ./output_images/heatmap.jpg
[image4]: ./output_images/heatmap_threshold.jpg
[image5]: ./output_images/detected_cars.jpg
[image6]: ./output_images/scales.jpg

[video1]: ./project_video.mp4


###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the IPython notebook section named "Histogram of Oriented Gradients (HOG)"

I started by reading in all the `vehicle` and `non-vehicle` images of the training set.

I converted each image to a feature space with the following steps:

1. convert color space from RGB to YCrCb.  All feature extractions listed below are performed in this color space
2. extract downscaled (32, 32) pixel color map
3. extract color histogram (32 bins)
4. extract HOG features in all color channels (9 orientations, 8 pixels per cell, 2 cells per block)


####2. Explain how you settled on your final choice of HOG parameters.

The combination of features extracted, their parameters, and colorspace chosen are the result of trial and error.  The objective was achieve highest accuracy score by the classifier.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in the IPython notebook section named "Training a Classifier"

Before feeding features of the training images to the classifier, I normalized the features using `StandardScaler`.  This computes a mean and std for each feature set and normalises them to a common scale.  The parameters used to normalize and scale are saved and used later for normalizing future test images.

I then split and shuffled training images to train and test sets with `train_test_split`.

I then ran linear SVM classifier and measured accuracy with the saved aside test set.
I reached over 99% accuracy.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The basics were researched in IPython section "Sliding window search".  A more efficient implementation that will be discussed here can be found in the IPython section "Using scale for efficiency".

In principle a sliding window search involves scanning a target region of the image, one patch at a time, looking for correct identification of a car.  Each image patch is tranformed to a feature set and evaulated with our trained classifier.  Performing a search naively works great but is inefficient because HOG feature extraction is an expensive operation and if our patches overlap, we essentially transform the pixels to HOG features more than once.

The steps are as follows:
1. scale image by a set factor (repeat for several scales).  The objective here is perform search on patches of different sizes.  Because we will do HOG subsampling, thsi ensures the result is easily broken into correctly sized blocks of HOG features.
2. Extract HOG features for the whole image (once!)
3. extract from result of step 2 the HOG features of a patch we are interested in
4. extract matching pixel patch and extract spatial and color histogram features for the patch
5. Combine HOG, spatial, and color histogram features and predict with trained SVM classifier. If car is detected, save patch coordinates for a successful detection.
6. Repeat steps 3-5 for all patches
7. Repeat steps 1-6 for different scales
8. Once all detected cars patches are saved, produce a heatmap
9. Apply threshold to the heatmap so weak signal detections are dropped.  Weak signal are expected for false positive where a car is 'detected' by few patches, most probably noise.
10. Use labels function to find boxes that encompases each 'blob' of heatmap pixels
11. Mark the label boxes on the original image a detection.

I used the test images to experiment with different scale and threshold values.  The results can be seen in the next section.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Compared to naive sliding window approach where each patch undergoes independant feature extraction, HOG subsampling described above is much faster.  This is due to the fact that the HOG features are extracted once for the whole image and then reused for each patch under question, thus allowing overlap without additional HOG feature extractions.

The image below shows prominent steps of the flow on test images.  This vualization fo multiple test images halped me to tweak values like which scales to use, threshold value, and classifier confidence

![alt text][image6]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Beyond the steps dicussed above, when it came to video feed, I added running average of detected cars.  Usign heatmap, I accumulated count of detections per pixel.  If a certain threshold is passed, I extract labels and mark the rectangle on the image as a detection. This smoothed out jerkiness and removed some false positive detections.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My implementation is not very robust but successfully demonstrates the basic techniques for car detection.  It will likely fail in these cases:

1. very small cars are not found.  These requires scanning very small windows which dramatically increases number of windows and procesing times. I could adapt search area to the window size so small window size is limited to middle of the image where cars are closer to the horizon.
2. False positive detections can crop in by passing heatmap threshold.  Further work is required to filter out these completely.
3. As can be see on the project video, when 2 cars are one behind the other, this algoritm does not distinguish between them well.

Challenges I faced include:

1. Image format conversions! Some images are jpg while others are png which mean different scales of pixel values.  This is hidden to the human eye but play havock on the classifier.
2. Working with sliding windows of different sizes was very intuitive and easy to understand.   Switching to single HOG feature extraction for the whole image was hard to grasp and debug.  The description of the class slide was not very easy for me to understand.

