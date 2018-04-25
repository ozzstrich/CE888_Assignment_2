from lime import lime_image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from copy_common import play_one_episode, play_n_episodes
from Initial_Code.train_atari import Model as taModel
from tensorpack import *
from PIL import Image
import tensorflow as tf
from skimage.segmentation import mark_boundaries

print "\n LIME Analysis starting"

data_folder = "Data/Breakout"
game_name = "/breakout"
data_path = 'Data/Breakout/breakout'

text_file = 'Data/Breakout/breakout.txt'
f = open(text_file)

model = play_one_episode

environment = "//Users/osama/Documents/University/Masters/CE888_Assignment/Assignment_2/Initial_Code/Breakout-v0.npz"


def get_pd(s, t):
    return model(s, t)

# IF NUM_ACTIONS Error: Set NUM_ACTIONS to number of actions available in game
# eg num_actions = 4, for breakout
predictor = OfflinePredictor(PredictConfig(
    model=taModel(),
    session_init=get_model_loader(environment),
    input_names=['state'],
    output_names=['policy']))

print "============= MODEL LODADED ============= \n"

nontrans_imgset = []
imgset = []
actset = []

# Change i to length of dataset as needed


def image_transform():
    for i in range(15):

        img = cv2.imread(data_folder + game_name + str(i) + '.png')
        nontrans_imgset.append(img)

        new_img = np.reshape(img, (1, 84, 84, 12))

        imgset.append(new_img)


        actions = int(f.readline())
        actset.append(actions)

        # Uncomment for predictions
        # (WARNING:  - Could OVERHEAT on large image set)
        # preds = get_pd(new_img, predictor)
        # print "Image", i, "prediction: ", preds, '\n'
    print "============= IMAGES TRANSFORMED ============= \n"


def lime_exp():
    print "============= LIME STARTING ============= \n"

    explainer = lime_image.LimeImageExplainer()
    # print "imgset length: ", len(imgset), '\n'

    # explain_instance(image, classifier_fn, labels=(1, ), hide_color=None, top_labels=5, num_features=100000, num_samples=1000, batch_size=10, qs_kernel_size=4, distance_metric='cosine', model_regressor=None)
    # for i in range(50):
    #    lime_img = np.reshape(imgset[i], (1, 84, 84, 3))
    # print "imgset 0: ", imgset[0]
    preds = get_pd(imgset[0], predictor)
    explanation = explainer.explain_instance(nontrans_imgset[0], preds, actset[0], hide_color = 0, num_samples = 1000)

    temp, mask = explanation.get_image_and_mask(295, positive_only = True, num_features = 5, hide_rest = True)
    fig = (mark_boundaries(temp / 2 + 0.5, mask))
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    fig.savefig("limeoutput.png")


image_transform()
lime_exp()
