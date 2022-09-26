import os
import requests
import numpy as np
from skimage.transform import resize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
# prevent annoying tensorflow warning

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import warnings
warnings.simplefilter("ignore")

CLASS_LABELS = ['ADONIS',
 'AFRICAN_GIANT_SWALLOWTAIL',
 'AMERICAN_SNOOT',
 'AN_88',
 'APPOLLO',
 'ATALA',
 'BANDED_ORANGE_HELICONIAN',
 'BANDED_PEACOCK',
 'BECKERS_WHITE',
 'BLACK_HAIRSTREAK',
 'BLUE_MORPHO',
 'BLUE_SPOTTED_CROW',
 'BROWN_SIPROETA',
 'CABBAGE_WHITE',
 'CAIRNS_BIRDWING',
 'CHECQUERED_SKIPPER',
 'CHESTNUT',
 'CLEOPATRA',
 'CLODIUS_PARNASSIAN',
 'CLOUDED_SULPHUR',
 'COMMON_BANDED_AWL',
 'COMMON_WOOD-NYMPH',
 'COPPER_TAIL',
 'CRECENT',
 'CRIMSON_PATCH',
 'DANAID_EGGFLY',
 'EASTERN_COMA',
 'EASTERN_DAPPLE_WHITE',
 'EASTERN_PINE_ELFIN',
 'ELBOWED_PIERROT',
 'GOLD_BANDED',
 'GREAT_EGGFLY',
 'GREAT_JAY',
 'GREEN_CELLED_CATTLEHEART',
 'GREY_HAIRSTREAK',
 'INDRA_SWALLOW',
 'IPHICLUS_SISTER',
 'JULIA',
 'LARGE_MARBLE',
 'MALACHITE',
 'MANGROVE_SKIPPER',
 'MESTRA',
 'METALMARK',
 'MILBERTS_TORTOISESHELL',
 'MONARCH',
 'MOURNING_CLOAK',
 'ORANGE_OAKLEAF',
 'ORANGE_TIP',
 'ORCHARD_SWALLOW',
 'PAINTED_LADY',
 'PAPER_KITE',
 'PEACOCK',
 'PINE_WHITE',
 'PIPEVINE_SWALLOW',
 'POPINJAY',
 'PURPLE_HAIRSTREAK',
 'PURPLISH_COPPER',
 'QUESTION_MARK',
 'RED_ADMIRAL',
 'RED_CRACKER',
 'RED_POSTMAN',
 'RED_SPOTTED_PURPLE',
 'SCARCE_SWALLOW',
 'SILVER_SPOT_SKIPPER',
 'SLEEPY_ORANGE',
 'SOOTYWING',
 'SOUTHERN_DOGFACE',
 'STRAITED_QUEEN',
 'TROPICAL_LEAFWING',
 'TWO_BARRED_FLASHER',
 'ULYSES',
 'VICEROY',
 'WOOD_SATYR',
 'YELLOW_SWALLOW_TAIL',
 'ZEBRA_LONG_WING']


class efficientNetB3:
    def __init__(self):
        filename = "EfficientNetB3-butterflies-0.97.h5"
        if not os.path.exists(filename):
            model_path = os.path.join("https://connectionsworkshop.blob.core.windows.net/butterflies", filename)
            r = requests.get(model_path)
            with open(filename, "wb") as outfile:
                outfile.write(r.content)
        self.model = tf.keras.models.load_model(filename)


    def predict(self, image: np.ndarray):
        ### resize all images to the size expected by the network
        image = resize(image, (150, 150),
                   preserve_range=True,
                   anti_aliasing=True)
        image = np.expand_dims(image, 0)
        result = self.model.predict(image)
        # find the highest weight, and corresponding class name and index
        max_index = 0
        max_result = ""
        max_weight = 0.
        for i, weight in enumerate(result[0]):
            if weight > max_weight:
                max_weight = weight
                max_result = CLASS_LABELS[i]
                max_index = i
        return "{}: {:.2f}%".format(max_result, max_weight*100)
