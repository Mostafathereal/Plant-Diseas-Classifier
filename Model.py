from keras import backend as k
from keras.applications.inception_v3 import InceptionV3
from keras.applications import InceptionResNetV2
from keras.models import Model, model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Dropout, Conv2D
from keras import optimizers

from keras import layers
from keras import models

train_path = 'Data/train'
test_path = 'Data/test'



base_model = InceptionV3(weights='imagenet', include_top = False, input_shape = (256, 256, 3))

for layer in base_model.layers[:301]:
    layer.trainable = False
 # print(base_model)
# x = base_model.output
base_model.compile(loss = "categorical_crossentropy", optimizer = optimizers.Adam(lr=0.0001), metrics=['accuracy'])

base_model.layers.pop()
base_model.compile(loss = "categorical_crossentropy", optimizer = optimizers.Adam(lr=0.0001), metrics=['accuracy'])
x = Conv2D(128, (1,1), activation='relu')(base_model.output)
x = Flatten()(x)
x = Dense(64, activation = "relu")(x)
x = Dropout(0.5)(x)
x = Dense(64, activation="relu")(x)

## predictions
x = Dense(15, activation="softmax")(x)



model = Model(input = base_model.input, output = x)

model.compile(loss = "categorical_crossentropy", optimizer = optimizers.Adam(lr=0.0001), metrics=['accuracy'])


train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size = (256,256), batch_size = 128)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size = (256,256), batch_size = 128)




model.summary()

model.fit_generator(train_batches, steps_per_epoch = 145, epochs = 5, verbose = 2)




# Data Augmentation
