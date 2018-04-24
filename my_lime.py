import comet_ml
from comet_ml import Experiment
import lime
from lime import lime_image
import sys
import os
import keras
from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import cv2
from copy_common import play_one_episode

from tensorpack import *
from tensorpack.utils.concurrency import ensure_proc_terminate, start_proc_mask_signal
from tensorpack.utils.serialize import dumps
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient
from tensorpack.utils.gpu import get_nr_gpu




print "\n LIME Analysis starting"

text_file = 'Data/Breakout/breakout.txt'
f = open(text_file)

model = play_one_episode

# model = inc_net.InceptionV3()
# mod = OfflinePredictor(PredictConfig(model=Model(), session_init=get_model_loader(args.load), input_names=['state'], output_names=['policy']))

environment = "/Initial_Code/Breakout-v0.npz"
output = []
actset = []


def get_pd(s, t):
    # for i in range(len(output)):
    #     s = output[i][i]
    #     print "s from PD",s
    #     print len(s)
    print " ======= S, T ====== "
    print s, t
    return model(s, t)


def trans_and_predict():

    for i in range(50):
        # img = image.load_img('Data/Breakout/breakout' + str(i) + '.png')
        # img = image.img_to_array(img)  # img to array might need to be int, not double
        # img = np.expand_dims(img, axis = 0)
        # img = preprocess_input(img)

        img = cv2.imread('Data/Breakout/breakout' + str(i) + '.png')

        # Revert image to array
        for it in range(4):
            img = image.img_to_array(img[:, :, it * 3:3 * (it + 1)])
            # print len(img)
            # print img.shape
            # print img
            output.append(img)
            # np.vstack(output)

        actions = str(f.readline())
        actset.append(actions)

        # print "output length: ", len(output), "\n"
        # # print "output shape: ", output[i].shape, "\n"
        # print "action from data: ", actset[i]
        # print "output: ", output[i], "\n"
        # preds = get_pd(output[i], actset[i])
        preds = get_pd(output[i][i][i][0], actset[i])
        print "PREDS: ", preds

        # print "\n ============= Image ", i + 1, " Prediction ============= \n"
        # for x in decode_predictions(preds)[0]:
        #     print x



def lime_exp():
    explainer = lime_image.LimeImageExplainer()

    # Hide color is the color for a superpixel turned OFF. Alternatively, if
    # it is NONE, the superpixel will be replaced by the average of its pixels

    # explain_instance(image, classifier_fn, labels=(1, ), hide_color=None, top_labels=5, num_features=100000, num_samples=1000, batch_size=10, qs_kernel_size=4, distance_metric='cosine', model_regressor=None)

    # My note: Get one explainer for now
    # for i in range(len(output)):
    explanation = explainer.explain_instance(output[0][0], actions, model, hide_color=0, num_samples=1000)

    from skimage.segmentation import mark_boundaries

    temp, mask = explanation.get_image_and_mask(295, positive_only=True, num_features=5, hide_rest=True)
    fig = (mark_boundaries(temp / 2 + 0.5, mask))
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    fig.savefig("output.png")


trans_and_predict()
# lime_exp()
