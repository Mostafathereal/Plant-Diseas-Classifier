# # import pandas as pd
# import numpy as np
# import keras
# import matplotlib.pyplot as plt
# from keras.applications import MobileNet
# from keras.preprocessing import image
# from keras.applications.mobilenet import preprocess_input
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Model
# from keras.optimizers import Adam

import json
import inception
from keras.applications.inception_v3 import InceptionV3
from keras.applications import InceptionResNetV2
from keras.models import Model, model_from_json
from keras.layers import Dense, GlobalAveragePooling2D

from keras import layers
from keras import models




base_model = InceptionV3(weights='imagenet', include_top = False)
 # print(base_model)

print(base_model.summary())
