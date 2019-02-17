data_type = "cail2018_big"  # cail_0518 | cail2018_big

raw_train_path = "./data/raw/cail_0518/data_train.json"  # small data
raw_valid_path = "./data/raw/cail_0518/data_valid.json"  # small data
raw_test_path = "./data/raw/cail_0518/data_test.json"    # small data

seg_train_path = "./data/raw/{}/data_train.json.seg".format(data_type)
seg_valid_path = "./data/raw/{}/data_valid.json.seg".format(data_type)
seg_test_path = "./data/raw/{}/data_test.json.seg".format(data_type)

raw_accu_path = "./data/raw/{}/accu.txt".format(data_type)
raw_law_path = "./data/raw/{}/law.txt".format(data_type)

big_raw_path = "./data/raw/cail2018_big/cail2018_big.json"

law_vocab_path = "./data/law.vocab"
stopwords_path = "./data/stopwords.txt"

