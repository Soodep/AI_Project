# CSCE-5210_AI_Class-Project
## Introduction
In this paper, we utilizes the CNNs approach for facial emotion recognition. We trained and tested our model on the FER2013 dataset which consists of 48x48 pixel grayscale images of faces. The images are format into a .csv file with 2 columns. The first column is consented of the labels and it has the different 'emotions' (0) angry, (1) disgusted, (2) fearful, (3) happy, (4) sad, (5) surprised, (6) neutral. The second column is the 'data' which are the different pixels in the images. The total count was 30000 images in our dataset.
![image](https://user-images.githubusercontent.com/78702377/116769307-b185f400-aa00-11eb-972a-1a88476c7476.png)


## Method
The hardware environment of this experiment is: 
Window10 operating system, Intel i7-7700KCPU, NVIDIA GeForce GTX 1080 Ti GPU hardware environment. 

The software environment is:
Python3.6.8 with PyTorch, and CUDA 11.1 is used to builda convolution neural network. 

Our network has three convolution layers and onefully connected layer.  Shown in following figure.  In the first convolution layer, we had 64 filters with kernelsize is 3x3. The second convolution is different fromfirst layer. In this layer, we have 128 3x3 filters. Thethird convolution layers has 256 3x3 filters.  These convolution layers also along with batch normaliza-tion, max-pooling layer and dropout. Pooling setupis 2x2 with a stride of 1 to reduce the size of the re-ceptive field and avoid overfitting. In dropout layer,a fraction of 0.2 is used.
![CNN architecture](https://user-images.githubusercontent.com/78702377/116769334-ec882780-aa00-11eb-82a5-32f1e8573b7e.png)

## Result
The initialization of model parameters is an important part in the process of neural network training. The appropriate initialization value can make the model converge quickly. The initialization of model parameters in this experiment is shown in following table. The batch size is 128, and the total number of training rounds is 200. The optimizer selects the random gradient descent algorithm (SGD), in which the learning rate parameter is set to 0.05, The decay value of learning rate after each update is set to 1e-5.

![image](https://user-images.githubusercontent.com/78702377/116769381-5274af00-aa01-11eb-9404-018d9fbcca6a.png)

Then we did several experiments by setting different parameters which showns on the following table.

![image](https://user-images.githubusercontent.com/78702377/116769457-e34b8a80-aa01-11eb-9155-b59ff8ebe400.png)


When Epoch is set to 100 and the optimizer is SGD, the accuracy of the training set is 99.79\%, the accuracy of the validation set is 58.58\%, and the loss value of the training set is 0.006. In order to improve the training The accuracy on the set and validation set needs to increase the number of training rounds, that means set Epoch to a larger value. When Epoch is set to 200, compared to when Epoch is set to 100, the accuracy of the training set is increased by 0.06\%, and the accuracy of the validation set is increased by 1.62\% which is 60.20\% and this is be highest accuray we got.

In order to verify whether other optimizers can improve the accuracy, we use Adam optimizer when other parameters are unchanged. When Epoch is set to 100, the accuracy of the training set is 24.77\%, and the accuracy of the verification set is 23.82\%. When Epoch is set to 200, the accuracy of the training set is 26.83\%, and the accuracy of the validation set is 25.71\%.

It can be seen from the data that when the Adam optimizer is used, no matter whether the epoch is increased or not, the final accuracy rate is very low, and the Loss rate value is also large. It is obvious that using the SGD optimizer is better than the Adam optimizer.

![image](https://user-images.githubusercontent.com/78702377/116769506-548b3d80-aa02-11eb-8574-ea948c8b2fa7.png)


## Reference
https://www.kaggle.com/msambare/fer2013

https://github.com/amineHorseman/facial-expression-recognition-using-cnn

https://blog.csdn.net/charzous/article/details/107452464?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-2&spm=1001.2101.3001.4242

https://www.overleaf.com/articles/convolutional-neural-networks-for-facical-emotion-recognition/grwnwbqvzmnr

https://www.cnki.com.cn/Article/CJFDTOTAL-DNZS202003091.htm
