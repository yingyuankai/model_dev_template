import pickle
import os
from collections import namedtuple
from configs.cfg_global import *
from deep_models.models.textcnn_model import Model
from deep_models.trainers.textcnn_trainer import Trainer
from deep_models.evalers.textcnn_evaler import Evaler
from deep_models.data_loader.data_generator import DataGenerator
from deep_models.data_processor.tfrecord_data_processor import DataProcessor
from deep_models.predictors.textcnn_predictor import Predictor
from deep_models.pack_model.pack_textcnn import PackModel
from utils.data_new import Data

exp_name = "textcnn"
train_sources = ['train']  # 'train', 'valid', 'test'
train_tasks = ['all']  # all/law/accu/impris
model_output_path = "./export/{}/{}".format(exp_name, "_".join(train_tasks))
log_root = 'log_{}'.format("_".join(train_tasks))
checkpoint_dir = '{}/ckpt'.format(log_root)
summary_dir = '{}/summary'.format(log_root)
# pretrain_embedding = "./export/fastext/cail2018_big/relevant_articles/1/fasttext.bin"
pretrain_embedding = "./data/wenshu/model.bin"
pretrain_word2vec_embedding = "./data/word2vec/fasttext.model"

op_prefix = 'prefix'

rebalance = True

data_base_path = "./data/{}/{}/".format(exp_name, data_type)
train_data_path = "{}/data.train{}.tfrecord".format(data_base_path, ".augmented" if rebalance else "")
valid_data_path = "{}/data.valid.tfrecord".format(data_base_path)
test_data_path = "{}/data.test.tfrecord".format(data_base_path)

# loss weight
raw_loss_weight_path = os.path.join(data_base_path, "loss_weight.pkl")
loss_weight_path = os.path.join(data_base_path, "loss_weight{}.pkl".format("_balance" if rebalance else ''))
if os.path.isfile(loss_weight_path):
    loss_weight_accu, loss_weight_law, loss_weight_impris, loss_weight_death = \
        pickle.load(open(loss_weight_path, "rb"))

synonym_path = "./data/word2vec/synonym.pkl"

use_pretrain_embedding = True

num_epochs = 100
eval_per_steps = 1000
learning_rate = 0.01
min_lr = 0.00000001
batch_size = 128
max_to_keep = 5
embed_size = 100
checkpoint_secs = 60
filter_sizes = [1, 2, 3, 5, 6, 7]
output_num_per_filter = 128
max_seq_len = 300
decay_steps = 100
decay_rate = 0.95
clip_gradients = 0.5
early_stopping = 10
eval_interval_secs = 60
keep_prob = 0.5
l2_lambda = 0.001

UNKNOWN_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"
