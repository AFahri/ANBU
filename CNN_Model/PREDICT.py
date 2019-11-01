#! /usr/bin/env python



import os
import sys

# add current directory to sys.path
# needed for Flask app to work
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)


from UTILS import get_root, load_pipeline, get_logger


ROOT_DIR = get_root()
MODEL_PATH = os.path.join(ROOT_DIR, 'assets', 'CNN_Model')
PREPROCESSOR_FILE = os.path.join(MODEL_PATH, 'preprocessor.pkl')
ARCHITECTURE_FILE = os.path.join(MODEL_PATH, 'gru_architecture.json')
WEIGHTS_FILE = os.path.join(MODEL_PATH, 'gru_weights.h5')


class PredictionPipeline(object):

    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, text):
        features = self.preprocessor.transform_texts(text)
        pred = self.model.predict(features)
        return pred


