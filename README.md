
## Introduction
This project is intented to create a model that predicts words inside captcha images  <br/>


![captcha example](images/b3xpn.png)
![captcha example](images/f228n.png)


## The model

The model i built make this steps to predict the words inside  images: <br/>

-Process the image and detect contours using OpenCV

![](images/contours.png)

-With the information provided by the contours, split the image in frames where each will contain a single character of the captcha text

![](images/chars.png)

- Then Run a convolutional neuronal network (CNN) created with Keras on each character to classify them

## Installation

Download first this repository on your local machine
```
git clone git@github.com:kun-ren/CaptchaCNN.git
cd CaptchaCNN
```
Copy datasets to ./dataset
Copy datasets used for evaluation to ./dataset_predict

Then you need to install dependencies
You can install anaconda and create a virtual environment:
```
conda create -n captcha_dl python=3.6 -y conda activate captcha_dl
```
And later activate the virtual environment to run any script
```
conda activate captcha-dl
pip install -r requirements.txt
```



## Running scripts

A couple of scripts are provided in the directory models/

The next code will print information relative to the dataset

```
cd models
```

if you want to train your own model:
```
python char_classifier.py --train --epochs 20 --batch-size 10 --test-size 0.1 
```
if you also want to evaluate your training result please use the command below instead:

```
python char_classifier.py --train --epochs 20 --batch-size 10 --test-size 0.1 --eval
```

you also can use pre-train model in this repository to predict directly:

```
python predict_img.py --image <Your-Captcha-Image-Path.bmp> --model ..\captcha_model.h5
```

