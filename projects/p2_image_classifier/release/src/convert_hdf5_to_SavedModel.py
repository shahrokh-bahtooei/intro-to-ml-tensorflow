import tensorflow as tf

import predict_flower

model = predict_flower.load_model('flower_classifier.h5')
model.save('flower_classifier_SavedModel')
