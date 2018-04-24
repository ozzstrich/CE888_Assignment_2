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

from copy_common import play_one_episode



print "\n Lime Analysis Starting \n"

f = open('Data/Breakout/breakout.txt')
actions = f.read().splitlines()

# Push info to comet
# experiment = Experiment(api_key="FVJK5r3DO7NBCS6Yt2S8WBu7z", project_name="ce888")

# =================== Get top 5 Predictions using Inception ===================
inet_model = inc_net.InceptionV3()


def transform_img_fn(path_list):  # Transforms image into array of numbers
    out = []
    for img_path in path_list:
        img = image.load_img(img_path)  # target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = inc_net.preprocess_input(x)
        out.append(x)
        print "x LEN: ", len(x)
        print x
    return np.vstack(out)

# Image prediction
# in range(number of images)
for i in range(5):
    images = transform_img_fn([os.path.join('Data/Breakout/breakout' + str(i) + '.png')])
    # I'm dividing by 2 and adding 0.5 because of how this Inception represents images
    plt.imshow(images[0] / 2 + 0.5)
    preds = inet_model.predict(images)
    # preds = get_pd(images, actions)
    print "\n"
    for x in decode_predictions(preds)[0]:
        print(x)

explainer = lime_image.LimeImageExplainer()

# Hide color is the color for a superpixel turned OFF. Alternatively, if
# it is NONE, the superpixel will be replaced by the average of its pixels
explanation = explainer.explain_instance(images[0], inet_model.predict, top_labels=5, hide_color=0, num_samples=1000)



from skimage.segmentation import mark_boundaries

temp, mask = explanation.get_image_and_mask(
    295, positive_only=True, num_features=5, hide_rest=True)
fig = (mark_boundaries(temp / 2 + 0.5, mask))
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
fig.savefig("output.png")
