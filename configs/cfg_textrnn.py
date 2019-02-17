import pickle
import os
from collections import namedtuple
from configs.cfg_global import *
from deep_models.models.textrnn_model import Model
from deep_models.trainers.textcnn_trainer import Trainer
from deep_models.evalers.textcnn_evaler import Evaler
from deep_models.data_loader.data_generator import DataGenerator
from deep_models.data_processor.tfrecord_data_processor import DataProcessor
from deep_models.predictors.textcnn_predictor import Predictor
from deep_models.pack_model.pack_textcnn import PackModel
from utils.data_new import Data


exp_name = "textrnn"
model_output_path = "/home/zhanglei/data/law_challenge_data/out_put"
train_sources = ['train']  # 'train', 'valid', 'test'
train_tasks = ['law']  # all/law/accu/impris
log_root = 'log_{}'.format("_".join(train_tasks))
checkpoint_dir = '{}/ckpt'.format(log_root)
summary_dir = '{}/summary'.format(log_root)
# pretrain_embedding = "./export/fastext/accusation/1/fasttext.bin"
pretrain_embedding = "/home/zhanglei/data/law_challenge_data/fasttext_model_as_pretrain_embedding_for_deep_model/cail2018_big/accusation/1/fasttext.bin"




op_prefix = 'prefix'

# data_base_path = "/home/zhanglei/data/law_challenge_data/tfrecord/cail_0518/"
data_base_path = "/home/zhanglei/data/law_challenge_data/tfrecord/{}/".format(data_type)
train_data_path = "{}/data.train.tfrecord".format(data_base_path)
valid_data_path = "{}/data.valid.tfrecord".format(data_base_path)
test_data_path = "{}/data.test.tfrecord".format(data_base_path)


# loss weight
if os.path.isfile(os.path.join(data_base_path, "loss_weight.pkl")):
    loss_weight_accu, loss_weight_law, loss_weight_impris = \
        pickle.load(open(os.path.join(data_base_path, "loss_weight.pkl"), "rb"))
use_pretrain_embedding = True


num_epochs = 100
eval_per_steps = 1000
learning_rate = 0.01
min_lr = 0.000001
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
l2_lambda = 0.01
hidden_size = 100 # hidden_size=enbed_size
UNKNOWN_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"

# num_classes = 20  #number of label
# learning_rate = 0.01
# batch_size = 1024
# decay_steps = 1000
# decay_rate = 0.9
# sequence_length = 100
vocab_size = 10000
# embed_size = 100
# is_training = True
# dropout_keep_prob = 1
# l2_lambda = 0.01
# num_epochs = 60 #embedding size
# validate_every = 1 #每10轮做一次验证
# use_embedding = True



print("input is {}".format("ok"))


