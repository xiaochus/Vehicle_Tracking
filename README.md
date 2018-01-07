# Vehicle_Tracking
OpenCV 3 &amp; Keras implementation of vehicles tracking with video data.

**Algorithm**  
1. Object Detection: **MOG2**  
2. Object Tracking: **KCF**  
3. Object Classification: **CNN**  

**Requirement**  
- Python 3.6  
- OpenCV 3.2 + contrib  
- Tensorflow-gpu 1.0  
- Keras 1.2  

## Data

We train our CNN model with MIT's vehicle and pedestrian data, click [here](https://pan.baidu.com/s/1qXRQ5dy) to download the original data and the processed data.

Video data is saved in the video folder.

## CNN Model

The CNN model we use is as follows：

![CNN](/image/model.png)

## Result
Run the following command to execute the program.

`python track.py --file "car.flv"`

![show](/image/show.png)
