import comet_ml
from comet_ml import Experiment
import lime
from lime import lime_image
import sys
import os
import keras
from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import cv2
from copy_common import play_one_episode

print "\n LIME Analysis starting"

text_file = 'Data/Breakout/breakout.txt'
f = open(text_file)
# model = play_one_episode.pd()
model = inc_net.InceptionV3()


def trans_and_predict():
    output = []

    for i in range(5):
        img = image.load_img('Data/Breakout/breakout' + str(i) + '.png')
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis = 0)
        img = inc_net.preprocess_input(img)
        output.append(img)

        # actions = f.readlines().splitlines()
        # preds = model.predict(img, actions)
        preds = model.predict(img)

        print "\n ============= Image ", i, " Prediction ============= \n"
        for x in decode_predictions(preds)[0]:
            print x



def lime_exp():
    explainer = lime_image.LimeImageExplainer()

    # Hide color is the color for a superpixel turned OFF. Alternatively, if
    # it is NONE, the superpixel will be replaced by the average of its pixels
    explanation = explainer.explain_instance(trans_and_predict().img[0], model.predict, top_labels=5, hide_color=0, num_samples=1000)

    from skimage.segmentation import mark_boundaries

    temp, mask = explanation.get_image_and_mask(295, positive_only=True, num_features=5, hide_rest=True)
    fig = (mark_boundaries(temp / 2 + 0.5, mask))
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    fig.savefig("output.png")

trans_and_predict()
lime_exp()
