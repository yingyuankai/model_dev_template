import os
import importlib
import tensorflow as tf
from utils.config import process_config
from utils.dirs import create_dirs
from test import Testor


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('config', 'cfg_han', 'path to configuration file')
tf.app.flags.DEFINE_string('mode', 'train', 'data/train/eval/pack_model/predict/test mode.')
tf.app.flags.DEFINE_string('io_mode', 'save', 'save/load mode.')
tf.app.flags.DEFINE_string('gpus', '2', 'cuda visible devices')

tf.logging.set_verbosity(tf.logging.WARN)
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(FLAGS.gpus)
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.7
gpu_config.gpu_options.allow_growth = True
gpu_config.allow_soft_placement = True


def train(config):
    # build data_obj
    data_obj = config.Data(config)
    setattr(config, "data_obj", data_obj)
    setattr(config, "mode", 'train')
    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create an instance of the model you want
    model = config.Model(config)
    # create trainer and pass all the previous components to it
    trainer = config.Trainer(model, config, gpu_config)
    # here you train your model
    trainer.train()


def eval(config):
    # build data_obj
    data_obj = config.Data(config)
    setattr(config, "data_obj", data_obj)
    setattr(config, "mode", 'valid')
    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create an instance of the model you want
    model = config.Model(config)
    # create trainer and pass all the previous components to it
    evaler = config.Evaler(model, config, gpu_config)
    # here you train your model
    evaler.eval()


def main(unused_args):
    config = importlib.import_module('configs.{}'.format(FLAGS.config))

    if FLAGS.mode == 'data':                  # 制作 tfrecord 数据
        data_processor = config.DataProcessor(config)
        data_processor.process(modes=['train']) # 'train', 'valid', 'test'
    elif FLAGS.mode == 'train':               # 训练
        train(config)
    elif FLAGS.mode == 'eval':                # 评估，与 train 一同启动监控模型在验证机上的性能并更新最好的模型
        eval(config)
    elif FLAGS.mode == 'pack_model':         # 打包模型pb
        data_obj = config.Data(config)
        setattr(config, "data_obj", data_obj)
        setattr(config, "mode", 'pack_model')
        setattr(config, "io_mode", FLAGS.io_mode)
        pack_mode = config.PackModel(config, gpu_config)
        pack_mode.process()
    elif FLAGS.mode == 'predict':
        # as example
        predictor = config.Predictor(config, gpu_config)
        results = predictor.predict(['南宁市江南区人民检察院指控，2014年9月12日14时许，被告人冯某去到南宁市江南区白沙北三里，'
                                     '趁无人之机，用螺丝刀撬开被害人谢某某停放在该处的一辆红旗牌电动车的电门锁，随后，'
                                     '冯某骑该车逃离现场。在驾车行至江南区江南大道西江码头附近时，冯某被公安人员抓获。'
                                     '被盗的电动车已被公安机关依法扣押并被返还给被害人。经南宁市价格认证中心鉴定，'
                                     '被盗的电动车案发时价值2550元。被告人冯某归案后如实供述了上述事实。'
                                     '被告人冯某曾因犯故意伤害罪于2008年8月6日被本院判处××，2010年3月6日刑满释放。'])
        print(results)
    elif FLAGS.mode == 'test':                # 测试，输出三个任务的score
        tester = Testor(config, gpu_config)
        score = tester.test()
        print(score)


if __name__ == '__main__':
    tf.app.run()
