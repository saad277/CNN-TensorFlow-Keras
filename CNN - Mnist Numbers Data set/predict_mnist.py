
import os

from os import environ 

from numpy import random
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical

import matplotlib.pyplot as plt


environ["TF_CPP_MIN_LOG_LEVEL"]="3"

#Loading Model

filepath=r"C:\Users\Saad-277\Desktop\NN\CNN - Mnist Numbers Data set\model_save.h5"

my_model=load_model(filepath,custom_objects=None,compile=True);


print(my_model.summary());

#Showing weight and biases

print("Last node Biases :");

print(my_model.get_weights()[-1]);

print("Last node weights :");

print(my_model.get_weights()[-2]);

#Loading MNIST dataset

(train_images,train_labels),(test_images,test_labels)=mnist.load_data();

rand_number=random.randint(0,1000);

rand_image=test_images[rand_number];

plt.imshow(rand_image,cmap="Greys");

plt.show();

#Predicting a Random Number Image
prediction=my_model.predict(rand_image.reshape(1,28,28,1),batch_size=1);

print("The Random number image generated is : {}".format(prediction));

#Evaluation test over dataset

test_images=test_images.reshape((10000,28,28,1));

test_labels=to_categorical(test_labels);

(eval_l,eval_acc)=my_model.evaluate(test_images,y=test_labels,batch_size=1000);

print("Evaluation Accuracy is : {:4.2f}".format(eval_acc*100));








