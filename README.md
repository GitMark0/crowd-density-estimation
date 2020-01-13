# Crowd Density Estimation

## Dataset links:
1) ShanghaiTech Crowd Counting Dataset : https://drive.google.com/file/d/16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI/view
2) Beijing BRT Dataset : https://github.com/XMU-smartdsp/Beijing-BRT-dataset


## Reviewed papers:
1) https://www.analyticsvidhya.com/blog/2019/02/building-crowd-counting-model-python/
2) https://gluon.mxnet.io/chapter14_generative-adversarial-networks/pixel2pixel.html
3) https://www.tensorflow.org/tutorials/generative/pix2pix
4) https://towardsdatascience.com/building-a-convolutional-neural-network-for-image-classification-with-tensorflow-f1f2f56bd83b
5) https://heartbeat.fritz.ai/a-research-guide-to-convolution-neural-networks-a4589b6ca9bd
6) https://www.researchgate.net/publication/285164623_An_Introduction_to_Convolutional_Neural_Networks

## How to run density estimator:
Start runner.py with one argument - number of the image from test_examples file as follows:
python runner.py image_number

## How to add new pictures:
Save target picture in test_examples/ in this format: 'IMG_*.jpg' where * stands for image number
