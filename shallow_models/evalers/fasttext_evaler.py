import os
from shallow_models.base.base_eval import BaseEval
from fastText import load_model
from utils.display_utils import print_results


class Evaler(BaseEval):
    def __init__(self, config):
        super(Evaler, self).__init__(config)

    def _eval_single_task(self, task):
        """

        :param task:
        :return:
        """
        model_path = os.path.join(self._config.model_output_path, task, str(int(self._config.multi_labels_in_one_line)))
        data_path = self._config.valid_data_path.format(self._config.data_base_path, task,
                                                        int(self._config.multi_labels_in_one_line))
        print("Eval for {}".format(model_path))
        model = load_model(os.path.join(model_path, "fasttext.{}".format(self._config.use_bin_or_ftz)))
        print_results(*model.test(data_path))

    def eval(self):
        """
        eval for each task
        :return:
        """
        for task in self._config.tasks:
            self._eval_single_task(task)
