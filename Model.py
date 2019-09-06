

import json
from keras.applications.inception_v3 import InceptionV3
from keras.applications import InceptionResNetV2
from keras.models import Model, model_from_json
from keras.layers import Dense, Flatten, Dropout, Conv2D

from keras import layers
from keras import models




base_model = InceptionV3(weights='imagenet', include_top = False, input_shape = (256, 256, 3))

for layer in base_model.layers[:300]:
    layer.trainable = False
 # print(base_model)
# x = base_model.output
x = Conv2D(128, (1,1), activation='relu')(base_model.output)
x = Flatten()(x)
x = Dense(128, activation = "relu")(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)

## predictions
x = Dense(2, activation="softmax")(x)



model = Model(input = base_model.input, output = x)

model.summary()
