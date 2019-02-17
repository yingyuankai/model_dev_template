import pickle
import os
from configs.cfg_global import *
from deep_models.predictors.ensemble_predictor import Predictor
from utils.data_new import Data

exp_name = "ensemble"
data_base_path = "./data/{}/{}/".format('textcnn', data_type)
model_names = ['textcnn', 'dpcnn']
model_weights = {"textcnn": 0.35, "dpcnn": 0.65}
model_output_path = "./export/{}/all"

op_prefix = 'prefix_{}'
max_seq_len = 300
batch_size = 128

UNKNOWN_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"
