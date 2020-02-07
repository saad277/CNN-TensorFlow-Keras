import tensorflow as tf

from tensorflow import keras
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten
from keras.utils import to_categorical

(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data();


x_train=x_train/255;
x_test=x_test/255;




model=Sequential();

model.add(Flatten(input_shape=(28,28)));

model.add(Dense(20,activation="relu"));
model.add(Dense(10,activation="softmax"));


model.summary();

model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"]);

model.fit(x_train,y_train,epochs=5);

#Save model

model.save(filepath=r"C:\Users\Saad-277\Desktop\NN\CNN - Mnist Fashion Data set\model_save.h5",overwrite=True);





print("xx");
