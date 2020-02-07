
import os

from os import environ 

from numpy import random
from keras.datasets import fashion_mnist
from keras.models import load_model
from keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np

environ["TF_CPP_MIN_LOG_LEVEL"]="3"

#Loading Model

filepath=r"C:\Users\Saad-277\Desktop\NN\CNN - Mnist Fashion Data set\model_save.h5"

my_model=load_model(filepath,custom_objects=None,compile=True);


print(my_model.summary());

#Showing weight and biases

print("Last node Biases :");

print(my_model.get_weights()[-1]);

print("Last node weights :");

print(my_model.get_weights()[-2]);

#Loading MNIST dataset

(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data();


print(train_images.shape)

#rand_number=random.randint(0,1000);

#rand_image=test_images[rand_number];

rand_image=test_images[2]

print(rand_image.shape)

plt.imshow(rand_image,cmap="Greys");

plt.show();

(test_loss,test_acc)=my_model.evaluate(test_images,test_labels,verbose=2);

print(test_acc);

predictions=my_model.predict(test_images);

print(predictions[1])

print(np.argmax(predictions[1]))


















