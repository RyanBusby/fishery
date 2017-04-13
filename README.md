
#### Image Processing / Classification

Given thousands of images coming from fishing boats, the Nature Conservancy seeks to automate the process of detecting illegal fishing. Visit [kaggle.com/c/the-nature-conservancy-fisheries-monitoring](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring) to read more.


Data available from  [https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/download/test_stg1.zip](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/download/test_stg1.zip)

'image_prep.py' prepares the images for processing by adjusting intensity of color channels, normalizing exposure, improving contrast, finds edges, and makes each image the same size without distorting the proportions. These steps remove some variability of the data so we can train on specific differences of the images.

'conv_net.py' trains a Convolution Neural Network on the image data.
