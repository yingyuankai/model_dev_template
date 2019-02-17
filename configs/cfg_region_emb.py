import pickle
import os
from collections import namedtuple
from configs.cfg_global import *
from deep_models.models.region_emb_model import Model
from deep_models.trainers.region_emb_trainer import Trainer
from deep_models.evalers.region_emb_evaler import Evaler
from deep_models.data_loader.data_generator import DataGenerator
from deep_models.data_processor.tfrecord_data_processor import DataProcessor
from deep_models.predictors.region_emb_predictor import Predictor
from deep_models.pack_model.pack_region_emb import PackModel
from utils.data_new import Data

# Data config
UNKNOWN_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"
rebalance = True

# Train config
exp_name = "region_embed"
train_sources = ['train']  # 'train', 'valid', 'test'
train_tasks = ['law']  # all/law/accu/impris
model_output_path = "./export/{}/{}".format(exp_name, "_".join(train_tasks))
log_root = 'log_{}_{}'.format("_".join(train_tasks), exp_name)
checkpoint_dir = '{}/ckpt'.format(log_root)
summary_dir = '{}/summary'.format(log_root)
pretrain_embedding = "./export/fastext/cail2018_big/relevant_articles/1/fasttext.bin"
pretrain_word2vec_embedding = "./data/word2vec/fasttext.model"
data_base_path = "./data/{}/{}/".format('textcnn', data_type)
train_data_path = "{}/data.train{}.tfrecord".format(data_base_path, ".augmented")
valid_data_path = "{}/data.valid.tfrecord".format(data_base_path)
test_data_path = "{}/data.test.tfrecord".format(data_base_path)
use_pretrain_embedding = True

# Model config
variant = 'WC'  # ['WC', 'CW', 'win_pool', 'scalar', 'multi_region']
region_merge_fn = "reduce_max" # reduce_max or reduce_sum or concat
region_size = 3
region_sizes = [3, 5, 7]
num_epochs = 100
eval_per_steps = 1000
learning_rate = 0.01
min_lr = 0.00000001
batch_size = 128
max_to_keep = 5
embed_size = 100
hidden_depth = embed_size * len(region_sizes) if variant == 'multi_region' else embed_size
checkpoint_secs = 60
output_num_per_filter = 128
max_seq_len = 300
decay_steps = 100
decay_rate = 0.95
clip_gradients = 0.5
early_stopping = 10
eval_interval_secs = 60
dropout = 0.5
l2_lambda = 0.001

# Predict config
op_prefix = 'prefix'