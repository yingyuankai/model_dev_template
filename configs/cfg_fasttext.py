"""Model and Hyperparameters for one run.

"""
from shallow_models.data_processor.seg_data_processor import DataProcessor as Seg_Data_Processor
from shallow_models.data_processor.fasttext_data_processor import DataProcessor
from shallow_models.trainers.fasttext_trainer import Trainer as Model
from shallow_models.evalers.fasttext_evaler import Evaler
from shallow_models.predictors.fasttext_predictor import Predictor
from .cfg_global import *


data_base_path = "./data/fasttext/{}/".format(data_type)
train_data_path = "{}/{}/data_{}.train"
valid_data_path = "{}/{}/data_{}.valid"
test_data_path = "{}/{}/data_{}.test"

char_train_data_path = "{}/{}/char_data_{}.train"
char_valid_data_path = "{}/{}/char_data_{}.valid"
char_test_data_path = "{}/{}/char_data_{}.test"

model_output_path = "./export/fastext/{}".format(data_type)
test_result_path = "./data/test/"

is_save_ftz = False
model_name = 'fasttest'
train_sources = ['train', 'valid']

# train
lr = 0.1                       # learning rate [0.1]
dim = 100                      # size of word vectors [100]
ws = 5                         # size of the context window [5]
epoch = 100                      # number of epochs [5]
min_count = 5                  # minimal number of word occurences [1]
min_count_label = 0            # minimal number of label occurences [0]
minn = 0                       # min length of char ngram [0]
maxn = 0                       # max length of char ngram [0]
neg = 5                        # number of negatives sampled [5]
word_ngrams = 1                # max length of word ngram [1]
loss = "softmax"               # loss function {ns, hs, softmax} [softmax]
bucket = 2000000               # number of buckets [2000000]
thread = 12                    # number of threads [12]
lr_update_rate = 100           # change the rate of updates for the learning rate [100]
t = 1e-4                       # sampling threshold [0.0001]
label = "__label__"            # labels prefix [__label__]
verbose = 2                    # verbosity level [2]
pretrained_vectors = ""        # pretrained word vectors for supervised learning []


# quantization
qout = False                   # quantizing the classifier [0]
cutoff = 0                     # number of words and ngrams to retain [0]
retrain = False                # finetune embeddings if a cutoff is applied [0]
dsub = 2                       # size of each sub-vector [2]
qnorm = False                  # quantizing the norm separately [0]

# data process
multi_labels_in_one_line = True
tasks = ['relevant_articles', 'accusation', 'term_of_imprisonment']
# tasks = ['term_of_imprisonment']

# predict | eval
return_n = 1
use_bin_or_ftz = 'bin'

# service
law_predict_port = 9119