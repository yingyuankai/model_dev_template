import pickle
import os
from collections import namedtuple
from configs.cfg_global import *
from deep_models.models.amtl_model import Model
from deep_models.trainers.amtl_trainer import Trainer
from deep_models.evalers.amtl_evaler import Evaler
from deep_models.data_loader.data_generator import DataGenerator
from deep_models.data_processor.tfrecord_data_processor import DataProcessor
from deep_models.predictors.textcnn_predictor_1 import Predictor
from deep_models.pack_model.pack_textcnn import PackModel
from deep_models.features.textcnn_feature import Feature
from utils.data_new import Data
import importlib

# Data config
UNKNOWN_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"
rebalance = True
use_label_loss_weight = True
use_pretrain_embedding = True
use_prior = False
use_wenshu_emb = False
# Train config
exp_name = "amtl"
train_sources = ['train', 'valid', 'test']  # 'train', 'valid', 'test'
train_tasks = ['all']  # all/law/accu/impris
model_output_path = "./export/{}/{}".format(exp_name, "_".join(train_tasks))
words_vocab_path = "./data/word.vocab.{}".format('big' if use_wenshu_emb else 'small')

pretrain_embedding = "./export/fastext/cail2018_big/relevant_articles/1/fasttext.bin"
wenshu_pretrain_embedding = "./data/wenshu/model100.bin"
pretrain_embedding = wenshu_pretrain_embedding if use_wenshu_emb else pretrain_embedding
pre_train_ckpt_dir = ""  # "log_all_dpcnn_2018062602"
pretrain_word2vec_embedding = "./data/word2vec/fasttext.model"
data_base_path = "./data/{}/{}/{}".format("textcnn", data_type, 'big_vocab/' if use_wenshu_emb else "")
train_data_path = "{}/data.train{}.tfrecord".format(data_base_path, ".augmented" if rebalance else "")
valid_data_aug_path = "{}/data.valid{}.tfrecord".format(data_base_path, ".augmented" if rebalance else "")
test_data_aug_path = "{}/data.valid{}.tfrecord".format(data_base_path, ".augmented" if rebalance else "")
valid_data_path = "{}/data.valid.tfrecord".format(data_base_path)
test_data_path = "{}/data.test.tfrecord".format(data_base_path)

# label loss weight

raw_loss_weight_path = os.path.join(data_base_path, "loss_weight.pkl")
loss_weight_path = os.path.join(data_base_path, "loss_weight{}.pkl".format("_balance" if rebalance else ''))
if os.path.isfile(loss_weight_path):
    loss_weight_accu, loss_weight_law, loss_weight_impris, loss_weight_death = \
        pickle.load(open(loss_weight_path, "rb"))

# Model config
num_epochs = 100
eval_per_steps = 1000
learning_rate = 0.0001  # 0.0001
min_lr = 0.00000001
batch_size = 128
max_to_keep = 5
filter_size = 3
channel_size = 250
num_repeats = 4
embed_size = 100
checkpoint_secs = 60
output_num_per_filter = 128
max_seq_len = 300
decay_steps = 1000
decay_rate = 0.95
clip_gradients = 0.5
early_stopping = 10
eval_interval_secs = 60
keep_prob = 0.5
l2_lambda = 0.0001
adv_loss_lambad = 0.05
diff_loss_lambad = 0.01
task_loss_weights = [0, 0, 1, 1]  # [1, 1, 0, 0] [0, 0, 1, 1]
adv = True
textcnn_config = importlib.import_module("configs.cfg_textcnn_1")
avgeraged = True
# Predict config
op_prefix = 'prefix'

log_root = 'log_{}_{}_{}_{}_{}_{}{}{}{}_{}'.format(
    "_".join(train_tasks),
    exp_name,
    str(learning_rate),
    str(l2_lambda),
    str(decay_steps),
    int(rebalance),
    int(use_label_loss_weight),
    int(use_prior),
    int(use_wenshu_emb),
    "".join([str(tmp) for tmp in task_loss_weights]))
checkpoint_dir = '{}/ckpt'.format(log_root)
summary_dir = '{}/summary'.format(log_root)