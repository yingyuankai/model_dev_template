import importlib
import tensorflow as tf
from test import Testor

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('config', 'cfg_fasttext', 'path to configuration file')
tf.app.flags.DEFINE_string('mode', 'test', 'seg_data/data/train/eval/predict/test mode.')


def main(unused_args):
    config = importlib.import_module('configs.{}'.format(FLAGS.config))
    if FLAGS.mode == 'seg_data':
        g_config = importlib.import_module('configs.cfg_global'.format(FLAGS.config))
        seg_data_processor = config.Seg_Data_Processor(g_config)
        # [train, valid, test] for small data, [split] for big data
        # seg_data_processor.process(['valid', 'test', 'train'])
        seg_data_processor.process(['split'])
    elif FLAGS.mode == 'data':
        data_processor = config.DataProcessor(config)
        data_processor.process(modes=['train', 'test', 'valid'])
    elif FLAGS.mode == 'train':
        model = config.Model(config)
        model.train()
    elif FLAGS.mode == 'eval':
        evaler = config.Evaler(config)
        evaler.eval()
    elif FLAGS.mode == 'predict':
        # as example
        predictor = config.Predictor(config)
        results = predictor.predict(['南宁市江南区人民检察院指控，2014年9月12日14时许，被告人冯某去到南宁市江南区白沙北三里，'
                           '趁无人之机，用螺丝刀撬开被害人谢某某停放在该处的一辆红旗牌电动车的电门锁，随后，'
                           '冯某骑该车逃离现场。在驾车行至江南区江南大道西江码头附近时，冯某被公安人员抓获。'
                           '被盗的电动车已被公安机关依法扣押并被返还给被害人。经南宁市价格认证中心鉴定，'
                           '被盗的电动车案发时价值2550元。被告人冯某归案后如实供述了上述事实。'
                           '被告人冯某曾因犯故意伤害罪于2008年8月6日被本院判处××，2010年3月6日刑满释放。'])
        print(results)
    elif FLAGS.mode == 'test':
        tester = Testor(config, None)
        score = tester.test()
        print(score)


if __name__ == '__main__':
    tf.app.run()
