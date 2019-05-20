# __Real-Time Face Recognition and Emotion Detection__
Machine Learning Final Project


## Team
_Yujie Hao_
_Yan Li_
_Xinyu Ma_
_Deliang Wang_


## Goal

The goal of this project is achieving the real-time face recognition and add sticker on the target's face which selected by target's emotions. The project have three parts: Face recognition, Emotions Detection and Sticker Selection.

The Face recognition used to find the area of targets' face and recognite who she/he is. The Emotions Detection used to find the emotion of the faces (We trained 7 emotions in our model). The Sticker Selection would draw a rectangle of the detected face, print the the owner's name and emotion, and add right sticker on the face. There will be no stickers if the emotion is neutral.


__Single People Detection:__
![Angry Face](https://i.imgur.com/xeZ2rVL.jpg)

__Multiple Real Time Face Detection and Emotion Recognition:__
![Mul Face](https://i.imgur.com/lpfV7bm.jpg)


## First Part: Achieved Real-Time Face Recognition
See [facerec.ipynb](https://github.com/JoyceHao/FinalProject/blob/master/Face_Recog/facerec.ipynb)

In this section: we firstly define a getTrainingData function to take pictures from the camera. When taking pictures, we need to make sure every angle of our faces are collected and the environment should not be too dark. We have four people in our group and we collected around 500 photos per person and about 1000 photos of others' face (which we extracted from lfw dataset). We resize the images into (128,128,3) and stored them in Dataset class and train_test_split them into training and testing data. In this way, we have totally 5 classes to predict.

Then we built a CNN model, use 'categorical_crossentropy' as loss function and Adam as optimizer. The Layers are as below:
![Mul Face](https://imgur.com/WyanwFo.jpg)
We train the model with 128 batch size and 6 epoch. We found both the training and testing accuracy is already close to 100% (99.5%) whne 6 epoch training is done.

Then we use the CascadeClassifier ('haarcascade_frontalface_alt2.xml') to detect the positions of faces in one single frame, get images of faces according to the positions imformation and use it as input of our trained CNN model to predict whose face is it. According to the predictions, we draw rectangles on the faces and print the name on frame2 which is deep copy of last frame. Constantly processing every frame and show frame2 on the  screen, we got real time face Recognition.


## Second Part: Dectected the Real-Time Face Emotions
See [emotion.ipynb](https://github.com/JoyceHao/FinalProject/blob/master/emotions/emotion.ipynb)

After we detected faces in a frame that captured from live camera, we can extract facial data from the whole frame and feed those face data into a pre-trained CNN emotion detection model and get emotion predictions for each face. For the emotion detection model, we used _fer2013_ dataset which contains 35K+ labeled faces in pixels.

There are 7 emotions in the dataset which labeled from 0 to 6, and image data saved in pixels column.
face data size: (35887, 48, 48, 1), label data size: (35887, 7)
_0: Angry 1: Disgust 2: Fear 3: Happy 4: Sad 5: Surprise 6: Neutral_
![dataframe](https://i.imgur.com/VBwKfPv.jpg=400x108)  

We loop through the dataframe to save face (X) and label (y) data into numpy arrays. Each face will be a 48*48 2D image. Labels are integers from 0 to 7. However, in order to train a CNN model with theses data, we need to expand the dimension of the input data and convert 1D label data into categorical data (like one hot coding style). 

Then the data will be split by the ratio of 25%.

The model we trained is a Keras linear model that contains multiple convolutional layers and pooling layers. For multi class data, the activation function we used are _Relu_ and _Softmax_. There are also multiple Batch Normalization layer and Dropout layer to speed up the training process and prevent overfitting. The output layer will have 7 output channels.  

After we test model with batch size 64 and 100 epochs, we get training accuracy around 87.4%  and validation accuracy around 63.8%.


## Third Part: Sticker Selection

_**Images Pre-processing**_
This step is used to get the face masks' data. Our model would detect 7 different emotions: angry, hate, fear, happy, sad, surprise and neutral which means that we need 7 face masks. We used OpenCV library to read and split each .png image in r, g, b, a channels. In next step, we merged the r,g and b channels to rgb_face. Then the rgb channels data and alpha channel data are saved in facial_dict.
Pseudocode:

    facial_dict = {}
    for i in range(7):
        face_img = read(facial_file[i])
        rgb_channels, alpha_channel = face_img
        facial_dict[emotion] = [rgb_channels, alpha_channel]
    
_**get_facecover method**_
The get_facecover method used to add a cover to the detected face. It required the left corner position (x,y), and the width and height(w,h) of target's face; the captured frame(img); the RGB image and alpha channel image (rgb_image, a) of the face mask.

Firstly, The get_facecover method would resize the face mask based on the target's face size. Secondly, the method would use the alpha channel to get the face mask without white space. Thirdly, the method would add the face mask on the top of the target face which means it changed the captured frame and return the frame.

    def get_facecover(x, y, w, h, img, rgb_image, a):
        # Resized sticker image based on resized h and w
        # Seted shifing offset
        # Find face
        # Generate Mask
        # resize alpha for future calculation
        # Get the area of face mask
        # Add face mask to the target area
        # Replace the original with the face mask
        # Return the edited frame
        return img

_**Main() Function**_
We use OpenCV to capture the real time frame and use our trained face recognition and emotion model to achieve the goal.

Final performance showed on [Demo](##Demo) Section.

**If you would like to run and check the final performance, just use the file   [facerec_direct.ipynb](https://github.com/joycehao/tmp.py) in the Face_Recog folder and run all the block directly.**


## Github Stucture:
    FinalProject/
        README.md
        Face_Recog/
            facerec_direct.ipynb
            facerec.ipynb
            *.png
            training_data_others/
            training_data_yellow/
            training_data_yifei/
            training_data_liang/
            training_data_me/
            Model/
                simple_CNN.530-0.65.hdf5
                facemodel.h5 (https://drive.google.com/file/d/1vuGpxYvRghCD9PDabWFKt8yT7yidgufK/view?usp=sharing)
        emotions/
            emotion_model.json
            emotion.ipynb
            emotion_model.h5 (https://drive.google.com/file/d/1LcF51IkLr-Bzgp8Zi-ar7vJ39NbVWdCl/view?usp=sharing)
            fer2013.csv (https://drive.google.com/file/d/1mJQm23NB0jsEHIOh2bFknbZU2Dkdfw9Z/view?usp=sharing)


## Demo (Youtube)
  
  Single Real Time Face Detection and Emotion Recognition: https://youtu.be/VpPWcxOsJms
  Multiple Real Time Face Detection and Emotion Recognition: https://youtu.be/-CLP04Yt-8Y


## Reference
https://docs.opencv.org/3.1.0/dd/d43/tutorial_py_video_display.html
https://github.com/Mjrovai/OpenCV-Face-Recognition
https://arxiv.org/pdf/1706.01509.pdf
https://medium.freecodecamp.org/facial-emotion-recognition-develop-a-c-n-n-and-break-into-kaggle-top-10-f618c024faa7
https://github.com/seed-fe/face_recognition_using_opencv_keras_scikit-learn
https://github.com/vipstone/faceai
