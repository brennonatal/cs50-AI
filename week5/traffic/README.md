# My process

In my first attempt, I organized my CNN in the same way that Brian showed in 
class with the MNIST dataset.
The accuracy of my model was around 5%, I tried to add more Convolutional and 
Pooling layers but the result remained the same.
Then I looked into Tensorflow CNN [tutorial](https://www.tensorflow.org/tutorials/images/cnn) 
and I saw that they normalize pixel values to be between 0 and 1. Dividing the image data by 255.
So I tried it and improved my model's accuracy to 35%!

At that point, I got stuck. I tried different numbers and sizes of layers, but my model wasn't improving.
So I decided to take a look at the images (I should have done this before), and my images were very messed up...
Turns out I resized my image incorrectly, so I looked in the OpenCV documentation and fixed my resize function.
My model achieved 98% accuracy right away! =) 
I kept trying different parameters for my model but 98% was the best I could reach without spending too much computational power/time.