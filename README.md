# How to Stitch Images in OpenCV

Image stitching is the process of combining multiple images that are taken from overlapping portions of the same scene into one bigger image. This is helpful for taking pictures of wider scenes that can’t be contained in one camera shot. This can also be used for creating higher resolution images.

This is a basic image stitching program of two overlapping photos using OpenCV. The procedure includes detecting feature points from each of the photos, comparing features from one photo to the features of the other to find best matches, finding an affine or homography mapping, warping the image to align it with the other, and finally, combing and blending them to obtain a single seamless looking picture.

<h2> Feature Detection </h2>

The program takes two images, the left and the right part of the panorama (see sample input images below). Then, it detects significant features in the images which will be candidates for matching later. An example of a good feature in an image is a corner, as there is significant changes in the pixel values within the neighborhood of a corner which makes it distinctive from other parts of the image.

<img src="https://user-images.githubusercontent.com/90839613/133882234-73aed51e-824e-4b10-97ed-fde307afb567.png" width="800">

One way of getting feature points is using Harris corner detector. As a prerequisite, the input images should first be converted to grayscale. Then there is an available function in OpenCV named “cornerHarris” that does the corner detection. This will produce a matrix with different intensity values, with the corners detected having the lighter intensities (higher values). The output has to be normalized first so the values of the resulting matrix will only range from 0 to 255. A sample result of this function can be seen below. The points in the output matrix of Harris corner detector that exceeded a certain intensity threshold are the ones chosen as features or keypoints.

Here are the results of Harris corner detector: (a) normalized result showing different intensity values, (b) corners from the previous image that are above threshold, called features, are marked, and (c) features are marked in the colored image for visualization

<img src="https://user-images.githubusercontent.com/90839613/133882340-d2bfca99-489a-4d7c-9e34-246e5e38a2bf.png" width="800">

After some experiments on different threshold levels, I used the value of 100 as this threshold gave enough number of keypoints and it produced the best final image. The lower the threshold value, the more the keypoints that will be selected.

Points alone wouldn’t make good descriptors for matching left and right images, as they only contain 1 pixel, and pixel colors are repeated throughout an image. To create descriptors, fixed size patches instead of just 1-px points should be used. In this step, I used the function “getRectSubPix” which retrieves the pixels within a patch of a certain size and x,y center. The size of the patch I used is 17x17 pixels and the center coordinates are from the keypoints from previous step. To know which patch size to use, I tested several patch sizes from 7x7 patch to 19x19 (partnered with different Euclidean distance thresholds) and got the best results using 17x17 patch size. The larger the patch size, the greater the distance between the matches becomes, but it makes the matching more accurate.

<h2> Matching Descriptors </h2>

In the previous section, we got a set of descriptors in the form of patches from both the left image and the right image. So far, these two sets are independent from each other. Now, at this section, the descriptors from the left image are matched to the right image. This is done by first flattening the patches from a square 3-channel matrix to 1-dimensional 1- channel vector. Then, the Euclidean distance between all the patches from the left image to all the patches from the right image are computed. The distance between the patches is representative of how similar they are and how likely that they are the same chunk of an object from the world scene.

I used the “norm” function from OpenCV to compute the Euclidean distance. This function takes 2 vectors (the flattened patches) and a type, in which I use “NORM_L2” to instruct the function to perform Euclidean distance operations.

For every left image, the smallest distance when paired thru all the right images is saved, as well as the index of the right image which gave the smallest distance. This left and right pair therefore is the closest match. Then this distance is further evaluated and compared over a distance threshold. If the distance between the left-right pair is less than the threshold, this match is considered a “good match” and will be included in the next steps. This filtering is done to further remove the descriptors that are not supposed to have actual matches, e.g. features not located on the overlapping sections of the two images.

Again, the threshold for this step was determined experimentally. The best distance threshold to use was determined to be 1700 as this produced the best final images. I tested threshold from 1000 to 2000, with 100-step increment. This threshold may look high, however, it is expected and appropriate as the patch size used in the experiment is a bit large so big color differences is expected.

<h2> Perpective Transformation </h2>

Once the good pairwise descriptors are determined, one of the image can now be transformed or warped to fit with the other image. To do this, the transformation matrix has to be identified. As mentioned earlier, there are 2 ways to do this: 1) Affine transformation or 2) Homography mapping.

Affine transformation involves combination of translation, rotation, scaling and/or shearing. It preserves points, straight lines and planes. It also preserves parallelism between two lines. On the other hand, homography maps the transformation of points from one plane’s perspective to another. It has no constraint on preserving parallelism, length, and angle.

<img src="https://user-images.githubusercontent.com/90839613/133882712-6235fd08-f25d-4647-8e41-4210c7728a9a.png" width="300">

For the affine transformation, I used the OpenCV function “estimateAffine3D”. This function computes the optimal affine transformation between two sets of 3D points using RANSAC algorithm. In summary, RANSAC works by trying out many different random subsets of the points and fitting them in a model. Those points that fit the model upto a specific allowable error threshold are called the inliers. If the number of inliers is big enough, it returns the estimated model.

The above function is specific for 3D points, however, this is still preferable to the other affine functions from OpenCV since the other does not use RANSAC and/or doesn’t report inliers. To be able to use this function, the 2D points acquired from the previous step should be converted to 3D points first. This can be done by extending the points into their homogenous coordinates by using the function “convertPointsToHomogeneous”. This function assigns “1” to the z- coordinate of each point on the vectors.

The first two parameters on the “estimateAffine3D” function are the homogenous points from the right image and from the left image, respectively. The right image is the first parameter since this is the one that is being transformed. Another important parameter for this function is the RANSAC threshold. This dictates how much re-projection error RANSAC allows to still consider a point as an inlier. For this parameter, I used the value 2 (default is 3) since this produced the best output image using this transformation.

The output of this function is the 3D affine transformation matrix, and a vector indicating if the point on that index is an inlier or an outlier. Note that this transformation matrix is a 3x4 matrix, which cannot be used in the warp OpenCV functions. To convert the 3x4 matrix to a 3x3 matrix, I added the values from the 3rd and 4th column in the matrix, and removed the 4th column. The 4th column is a translation vector, and adding them to the 3rd column (z-axis vector) will contribute to the translation of the new x, y, and z coordinates by scale of 1.

For the homography mapping, I used the OpenCV function “findHomography”. The parameters needed in this function are the points of the descriptors from the image plane to be projected, the points of the corresponding descriptors from the target plane, the method used in computing the homography matrix, and the re-projection error threshold. Again the keypoints from the right image goes to the first parameter while the left to the second. Method selected for finding the homography is RANSAC, and the threshold used is 4, since the default value 3 does not produce a very good fit. The output, similar with the first method, is a transformation matrix with size 3x3, and the inlier/outlier vector.

Some statistics from the RANSAC results by both transformation methods are shown in the table below. The average residual computed using the sample pictures through affine is 3.34426 while 1.2381 through homography. These values represents the goodness of fit of the transformed right image to the left image. To compute this, I first used “perspectiveTransform” function to project the right keypoints to the perspective of the left image using either affine or homography transformation matrix. Then for every inlier pairs, I computed for the squared distance between the point coordinates in left image and the corresponding transformed coordinates in the right image. Finally, the average from these individual residuals is computed.

<table>
  <tr>
    <th>DESCRIPTION</th>
    <th>AFF</th>
    <th>HOM</th>
  </tr>
  <tr>
    <th>Total # of good pairwise descriptors</th>
    <td>518</td>
    <td>518</td>
  </tr>
  <tr>
    <th>RANSAC threshold used</th>
    <td>2</td>
    <td>4</td>
  </tr>
  <tr>
    <th># of descriptors considered as inliers</th>
    <td>183 (35.3%)</td>
    <td>189 (36.5%)</td>
  </tr>
  <tr>
    <th># of descriptors considered as outliers</th>
    <td>335 (64.7%)</td>
    <td>329 (63.5%)</td>
  </tr>
  <tr>
    <th>Average residual</th>
    <td>3.34426</td>
    <td>1.2381</td>
  </tr>
</table>

The inlier matches produced by RANSAC for homography mapping is displayed below:

<img src="https://user-images.githubusercontent.com/90839613/133883718-8d17f1b4-ac7a-496f-9caf-716aabf1e72c.jpg" width="800">

After determining the transformation matrix from either method, the right image can now be warped or transformed. This is done so the two images can finally be placed beside each other and fit. It is important to use the right image here and not the left since it is the first parameter used in the functions from affine and homography above. The function used here is “warpPerspective”. The transformation matrix that is returned by either of the previous function is the input needed for warping. Another important parameter I used here is the border mode (or the method for pixel interpolation) called “BORDER_REPLICATE”. The other option and the default mode is “BORDER_CONSTANT” which uses black pixels to fill in the blank parts of the warped image. The “replicate” mode uses the pixel value from the border and repeats them to fill up the blank parts. I found this useful since the blending of images (discussed in the next section) is better with the “replicate” method. The other method produces a dark gray line near the warped image’s border due to the black background after pixels from the 2 images were blended.

<h2>Stitching and Post-Processing</h2>

Simply placing the two images mentioned above beside each other does not show continuity as there are color differences between the two images (e.g. difference in brightness/illumination) and a line dividing the two sections becomes evident. The image that is put together without blending is shown below. 

<img src="https://user-images.githubusercontent.com/90839613/133883215-8ac34f4f-77c1-4329-9de1-fcb00173a79b.png" width="800">

To solve this and further improve the stitched image, I blended the pixels on the overlapping portion of the two images. To blend the image and remove the seams, I used weighted averaging on the overlapping pixels. The weight assigned is dependent on how far away a certain pixel is from the left or right image, horizontally.

We can define the overlapping area as the rectangle bounded by the left-most side of the warped right image, and the right-most side of the left image. The second’s x-value can be easily obtained from the left image matrix’s column’s value. The first can be identified by using again the function “perspectiveTransform”. The top left corner of the right image has coordinate (0,0). Transforming this using the transformation matrix from previous section, we can get the corresponding value of the top-left coordinate in the final image. Once we define these two x values, we can estimate the overlapping region. Everything else outside this region will use exclusively either the left image’s pixels or right image’s pixels, whichever is more appropriate.

The next step is identifying the color to use in each pixel within the overlapping area using weighted averaging. This weighted averaging allows a gradual change from the colors of the left image to the right image horizontally.

In addition to the color blending, it is also better to crop the image to only display the actual scene and remove the parts that are just created to fill in the blank spaces during warping. To identify in which x-coordinate the image should be cropped, I used again the “perspectiveTransform” function, same as with the procedure described earlier in this section. Since we know the coordinates of the top-right corner from original right image, we can compute for the coordinates of the top-right corner in the final image using the transformation matrix. We crop at its x-value.

The final image is displayed below:

Using affine transformation:

<img src="https://user-images.githubusercontent.com/90839613/133883337-1a3f00cd-aef9-45e0-a844-8bdcd62697d1.jpg" width="800">

Using homography mapping:

<img src="https://user-images.githubusercontent.com/90839613/133883379-1c462bb5-0189-4153-81f4-db255e8625e5.jpg" width="800">

