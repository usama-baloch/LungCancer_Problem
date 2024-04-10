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
