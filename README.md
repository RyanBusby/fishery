
## Image Processing / Classification

Given thousands of images, the Nature Conservancy seeks to automate the process of detecting illegal fishing. The images are received from cameras mounted on fishing boats. The occurrence of illegal fishing is detected by knowing what species of fish are present in an image. Visit [kaggle.com/c/the-nature-conservancy-fisheries-monitoring](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring) to read more.


Data available from  [https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/download/test_stg1.zip](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/download/test_stg1.zip)

First the images are adjusted using the scikit-image library, making use of filters that increase the contrast and adjust exposure. Then the images are all made to be the same size.

The type of filtering done on the images was determined based on exploration, shown in image_eda.ipynb. structure_data.py collects all of the training data and applies the image processing techniques and creates one dataset for training.

A convolutional neural network is then trained on the images. First images that received no adjustments are used to train a model as a baseline, and then the images that were adjusted are used to train another model, for improved accuracy in determining the species of fish present in the photo.
