import pickle
import os
from collections import namedtuple
from configs.cfg_global import *
from deep_models.models.textcnn_model_1 import Model
from deep_models.trainers.textcnn_trainer import Trainer
from deep_models.evalers.textcnn_evaler import Evaler
from deep_models.data_loader.data_generator import DataGenerator
from deep_models.data_processor.tfrecord_data_processor import DataProcessor
from deep_models.predictors.textcnn_predictor_1 import Predictor
from deep_models.pack_model.pack_textcnn import PackModel
from deep_models.features.textcnn_feature import Feature
from utils.data_new import Data

rebalance = True
use_label_loss_weight = True
use_pretrain_embedding = True
use_prior = False
use_wenshu_emb = False

exp_name = "textcnn"
train_sources = ['train', "valid", 'test']  # 'train', 'valid', 'test'
train_tasks = ['all']  # all/law/accu/impris

fasttext_pretrain_embedding = "./export/fastext/cail2018_big/relevant_articles/1/fasttext.bin"
wenshu_pretrain_embedding = "./data/wenshu/model100.bin"
pretrain_embedding = wenshu_pretrain_embedding if use_wenshu_emb else fasttext_pretrain_embedding
pre_train_ckpt_dir = ""  # "log_all_textcnn_2018062801"
op_prefix = 'prefix'
synonym_path = "./data/word2vec/synonym.pkl"
words_vocab_path = "./data/word.vocab.{}".format('big' if use_wenshu_emb else 'small')

task_loss_weights = [1, 1, 0, 0]  # [0, 0, 1, 1] [1, 1, 0, 0]
data_base_path = "./data/{}/{}/{}".format(exp_name, data_type, 'big_vocab/' if use_wenshu_emb else "")
model_output_path = "./export/{}/{}".format(exp_name, "_".join(train_tasks))
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

num_epochs = 100
eval_per_steps = 1000
learning_rate = 0.001  # 0.01
min_lr = 0.00000001
batch_size = 128
max_to_keep = 5
embed_size = 100
checkpoint_secs = 60
filter_sizes = [1, 2, 3, 5, 6, 7]
output_num_per_filter = 512
max_seq_len = 300
decay_steps = 1000
decay_rate = 0.95
clip_gradients = 0.5
early_stopping = 10
eval_interval_secs = 60
keep_prob = 0.5
l2_lambda = 0.00001

impris_mean = 15.43708183221214
impris_std = 24.26420000525387

avgeraged = False

log_root = 'log_{}_{}_{}_{}_{}_{}_{}{}{}{}_{}'.format(
    "_".join(train_tasks),
    exp_name,
    str(learning_rate),
    str(l2_lambda),
    str(decay_steps),
    str(output_num_per_filter),
    int(rebalance),
    int(use_label_loss_weight),
    int(use_prior),
    int(use_wenshu_emb),
    "".join([str(tmp) for tmp in task_loss_weights]))
checkpoint_dir = '{}/ckpt'.format(log_root)
summary_dir = '{}/summary'.format(log_root)

UNKNOWN_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"
