# LungCancer_Problem

### Problem Statement:

I worked on one of the most important topics which is the detection and classification of cancer in lungs. there are multiple steps to perform to get
the result which is to detect and classify lung tumors. there are two types of lung tumors(malignant and benign), the benign is the tumor which means 
the patient doesn't have any cancer, and malignant means it has cancer.

### Dataset:

-> We have a dataset that contains CT Scans which are essentially 3D X-rays, represented as a 3D Array of single-channel data, this is like a
stacked set of grayscale PNG images.

-> Just as a pixel exists within a two-dimensional space (like the rows and columns of a digital image),
a voxel exists within a three-dimensional space. This space is defined by three axes: length, width, and height.

-> CT scans utilize X-ray technology, but instead of a single projection, they involve taking multiple X-ray images from different angles around the body.

-> These multiple X-ray images are then reconstructed using computer algorithms to create cross-sectional slices or three-dimensional representations of the internal structures

#### What is Nodule?

Here: a Nodule is a solid mass of tissue that forms when abnormal cells group in the lung in a tumor. A tumor can be benign or malignant.
If it is benign then there is no cancer but if it is malignant then it is.  A small tumor in the lung (just a few millimeters wide) is called a nodule.
40% of lung nodules turn out to be malignant.

#### In-depth details of Dataset:

-> The CT Files have two files, one is .mhd file which contains the metadata of the image and one is .raw file containing the raw bytes
that make up the 3D Array.

-> Each File name starts with a unique series UID (the name comes from the DICOM) for the CT scan. For example, for series 1.2.3 the two files would be
1.2.3.mhd, 1.2.3.raw.

-> The candidates.csv file contains information about all lumps that potentially look like
nodules, whether those lumps are malignant, benign tumors, or something else altogether.

-> So we have 551,000 lines, each with a series-uid (which we’ll call series_uid in the
code), some (X,Y,Z) coordinates, and a class column that corresponds to the nodule
status (it’s a Boolean value: 0 for a candidate that is not an actual nodule, and 1 for a
candidate that is a nodule, either malignant or benign). We have 1,351 candidates
flagged as actual nodules.

-> The annotations.csv file contains information about some of the candidates that
have been flagged as nodules. We have size information for about 1,200 nodules.
This is useful since we can use it to make sure our training and validation data includes a
representative spread of nodule sizes. Without this, it’s possible that our validation set
could end up with only extreme values, making it seem as though our model is underperforming

#### Link of the Dataset: https://luna16.grand-challenge.org/
#### Alternative Link of the Dataset if the above doesn't work: https://zenodo.org/records/3723295


### High level Steps for the Solution:

Here is the end-to-end full system to find out the patient's lung cancer using the CT Scan data, There are five main steps:


1) Load the CT data file to take ct instance that is in the form of a 3D Scan. We use PyTorch to do all the conversion and loading.

2) Feed the CT Scan data into a module that performs segmentation which flags the voxels of interest,
Identify the voxels of potential tumors in the lungs using PyTorch to implement a technique known as segmentation. This is roughly akin to producing a
heatmap of areas that should be fed into our classifier in step 3. This will allow us to focus on potential tumors inside the lungs and
ignore huge swaths of uninteresting anatomy (a person can’t have lung cancer in the stomach, for example)

3) group the interesting voxels into small lumps in the search for candidate nodules. Here, we will find the rough center of each
hotspot on our heatmap. Each nodule can be located by the index, row, and column of its center point.

4) The nodule locations (index, row, column) are combined back with the CT voxel data to produce nodule candidates, which can then be examined by
our nodule classification model to determine whether they are actually nodules in the first place or not. we use 3D Convolution for classification.

5) and after step 4 when we find out that this voxel group is a nodule then we find whether it is malignant or not.
Similar to the nodule classifier in the previous step, we will attempt to determine whether the nodule is benign or malignant based on imaging data alone. We
will take a simple maximum of the per-tumor malignancy predictions, as only one tumor needs to be malignant for a patient to have cancer


-> The data we’ll use for training provides human-annotated output for both steps 3 and 4
-> Human experts have annotated the data with nodule locations, so we can work on either steps 2 and 3 or step 4 in whatever order we prefer

-> Our segmentation model is forced to consume the entire image, but we will structure things so that our classification model gets a
zoomed-in view of the areas of interest.

-> CTs order their slices such that the first slice is the inferior (toward the feet). So, Matplotlib renders the images upside
down unless we take care to flip them

-> the size of a nodule to 3 cm or less, the higher sizes will be called lung mass, not lung nodules.

-> The key part is this: the cancers that we are trying to detect will always be nodules, either suspended in the very non-dense tissue of the lung or
attached to the lung wall. That means we can limit our classifier to only nodules, rather than have it examine all tissue.
