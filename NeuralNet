#############################################################

import numpy as np 
import os 
import cv2 

print("hey...")
DATADIR="C:/Users/Alfred/Desktop/dinasour/kpwrt"
CATEGORIES=["success","failure"]

IMG_SIZE=50

training_data=[]

def create_training_data():
    count=0
    for category in CATEGORIES:
        path=os.path.join(DATADIR,category)
        class_num=CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
                count=count+1
                print(count)
            except Exception as e:
                pass
    
create_training_data()
print(len(training_data))

##############################################################

import random 

random.shuffle(training_data)
print("shuffling done...")


###############################################################

for sample in training_data[:30]:
    print(sample[1])
	
###############################################################


X=[]
y=[]

for features,label in training_data:
    X.append(features)
    y.append(label)
	
print(X[1])
X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
print(X[1])


#################################################################


import pickle

pickle_out=open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out=open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()


##################################################################


import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D


X=pickle.load(open("X.pickle","rb"))
y=pickle.load(open("y.pickle","rb"))

X=X/255.0

model=Sequential()
model.add(Conv2D(64,(3,3),input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])

model.fit(X,y,batch_size=2,epochs=10,validation_split=0.1)

######################################################################

model.save("dinasour.model")

#######################################################################
