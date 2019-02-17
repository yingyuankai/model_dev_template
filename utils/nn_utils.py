import tensorflow as tf
import os
import six
from tensorflow.python.ops import variable_scope, array_ops
from tensorflow.python.framework import ops
from tensorflow.contrib import slim
from tensorflow.contrib import rnn
from tensorflow.contrib.framework import list_variables, load_checkpoint
import numpy as np


def cal_single_label_accuracy(logits, labels, name=""):
    """

    :param logits:
    :param labels:
    :param name:
    :return:
    """
    pre = tf.argmax(logits, 1, name="predict_{}".format(name))
    real = labels
    correct_prediction = tf.equal(tf.cast(pre, tf.int32), tf.cast(real, tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='single_label_accuracy_{}'.format(name))
    return accuracy


def cal_regess_accuracy(logits, labels, mean, std, name=""):
    # labels_normlized = tf.divide(tf.subtract(tf.cast(labels, tf.float32), mean), std)
    labels_normlized = tf.cast(labels, dtype=tf.float32)
    logits = tf.squeeze(logits)
    accuracy = tf.abs(logits - labels_normlized)
    accuracy = tf.reduce_mean(accuracy, name=name)
    return accuracy


def cal_multi_label_accuracy(logits, labels, name=""):
    """
    calculate multi label accuracy
    :param logits:
    :param labels:
    :param name:
    :return: avg_accuracy, exact_accuracy
    """
    pre = tf.argmax(logits, 1, name="predict_{}".format(name))
    pre_one_hot = tf.one_hot(pre, depth=tf.shape(labels)[1])
    correct_prediction = tf.multiply(pre_one_hot, tf.cast(labels, tf.float32))
    all_labels_true = tf.reduce_max(tf.cast(correct_prediction, tf.float32), 1)
    # accuracy where all labels need to be correct
    accuracy = tf.reduce_mean(all_labels_true, name='multi_label_accuracy_{}'.format(name))
    return accuracy


def metric_variable(shape, dtype, validate_shape=True, name=None):
    """
    Create variable in 'GraphKeys.(LOCAL|METRIC_VARIABLES) collections.'
    :param shape:
    :param dtype:
    :param validate_shape:
    :param name:
    :return:
    """
    return variable_scope.variable(
        lambda: array_ops.zeros(shape, dtype),
        trainable=False,
        collections=[ops.GraphKeys.LOCAL_VARIABLES],
        validate_shape=validate_shape,
        name=name,
    )


def streaming_counts(y_true, y_pred, num_classes):
    # Weights for the weighted f1 score
    weights = metric_variable(
        shape=[num_classes], dtype=tf.int64, validate_shape=False, name='weights'
    )

    # Counts for the macro f1 score
    tp_mac = metric_variable(
        shape=[num_classes], dtype=tf.int64, validate_shape=False, name="tp_mac"
    )

    fp_mac = metric_variable(
        shape=[num_classes], dtype=tf.int64, validate_shape=False, name="fp_mac"
    )

    fn_mac = metric_variable(
        shape=[num_classes], dtype=tf.int64, validate_shape=False, name="fn_mac"
    )
    # Counts for the micro f1 score
    tp_mic = metric_variable(
        shape=[], dtype=tf.int64, validate_shape=False, name="tp_mic"
    )
    fp_mic = metric_variable(
        shape=[], dtype=tf.int64, validate_shape=False, name="fp_mic"
    )
    fn_mic = metric_variable(
        shape=[], dtype=tf.int64, validate_shape=False, name="fp_mic"
    )

    # Update ops for the macro f1 score
    up_tp_mac = tf.assign_add(tp_mac, tf.count_nonzero(y_pred * y_true, axis=0))
    up_fp_mac = tf.assign_add(fp_mac, tf.count_nonzero(y_pred * (y_true - 1), axis=0))
    up_fn_mac = tf.assign_add(fn_mac, tf.count_nonzero((y_pred - 1) * y_true, axis=0))

    # Update ops for the micro f1 score
    up_tp_mic = tf.assign_add(tp_mic, tf.count_nonzero(y_pred * y_true, axis=None))
    up_fp_mic = tf.assign_add(fp_mic, tf.count_nonzero(y_pred * (y_true - 1), axis=None))
    up_fn_mic = tf.assign_add(fn_mic, tf.count_nonzero((y_pred - 1) * y_true, axis=None))

    # Update op for the weights, just summing
    up_weights = tf.assign_add(weights, tf.reduce_sum(y_true, axis=0))

    # Grouping values
    counts = (tp_mac, fp_mac, fn_mac, tp_mic, fp_mic, fn_mic, weights)
    updates = tf.group(up_tp_mic, up_fp_mic, up_fn_mic, up_tp_mac, up_fp_mac, up_fn_mac, up_weights)
    return counts, updates


def streaming_f1(counts):
    # uppacking values
    tp_mac, fp_mac, fn_mac, tp_mic, fp_mic, fn_mic, weights = counts

    # normalize weights
    weights /= tf.reduce_sum(weights)

    # computing the micro f1 score
    prec_mic = tp_mic / (tp_mic + fp_mic)
    rec_mic = tp_mic / (tp_mic + fn_mic)
    f1_mic = 2 * prec_mic * rec_mic / (prec_mic + rec_mic)
    f1_mic = tf.reduce_mean(f1_mic)

    # computing the macro and weighted f1 score
    prec_mac = tp_mac / (tp_mac + fp_mac)
    rec_mac = tp_mac / (tp_mac + fn_mac)
    f1_mac = 2 * prec_mac * rec_mac / (prec_mac + rec_mac)
    f1_wei = tf.reduce_sum(f1_mac * weights)
    f1_mac = tf.reduce_mean(f1_mac)

    return f1_mic, f1_mac, f1_wei


def tf_f1_score(y_true, y_pred):
    f1s = [0, 0, 0]
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)

    for i, axis in enumerate([None, 0]):
        TP = tf.count_nonzero(y_pred * y_true, axis=axis)
        FP = tf.count_nonzero(y_pred * (y_true - 1), axis=axis)
        FN = tf.count_nonzero((y_pred - 1) * y_true, axis=axis)

        precision = tf.cast(TP, tf.float64) / (tf.cast(TP + FP, tf.float64) + 1e-12)
        recall = tf.cast(TP, tf.float64) / (tf.cast(TP + FN, tf.float64) + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        f1s[i] = tf.reduce_mean(f1)

    weights = tf.reduce_sum(y_true, axis=0)
    weights /= tf.reduce_sum(weights)
    f1s[2] = tf.reduce_sum(f1 * weights)
    micro, macro, weighted = f1s
    return micro, macro, weighted


def cal_double_multi_label_score(logits, labels, num_classes, name=""):
    logits_reshaped = tf.reshape(logits, shape=[-1, 2, num_classes])
    logits_argmax = tf.argmax(logits_reshaped, axis=1)
    logits_pre = tf.cast(tf.equal(logits_argmax, 0), tf.int64)

    return tf_f1_score(labels, logits_pre)


def cal_multi_label_score(logits, labels, name=""):
    logits_sig = tf.nn.sigmoid(logits)
    logits_threshold = tf.greater(logits_sig, 0.5)
    logits_pre = tf.cast(logits_threshold, tf.int64)

    return tf_f1_score(labels, logits_pre)


def loss_single_label(logits, labels, loss_weight=None):
    with tf.name_scope("single_label_loss"):
        if loss_weight is not None:
            class_weights = tf.gather(loss_weight, labels)
            losses = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, weights=class_weights)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(losses)
    return loss


def loss_regress(logits, labels, mean, std, loss_weight=None):
    with tf.name_scope("regess_loss"):
        # labels_normlized = tf.divide(tf.subtract(tf.cast(labels, tf.float32), mean), std)
        labels_normlized = tf.cast(labels, dtype=tf.float32)
        losses = tf.divide(tf.pow((logits - labels_normlized), 2), 100.)
        if loss_weight is not None:
            class_weights = tf.gather(loss_weight, labels)
            losses = losses * class_weights
        loss = tf.reduce_mean(losses)
    return loss


def loss_regress_new(logits, labels, mean, std, loss_weight=None):
    with tf.name_scope("regess_loss"):
        logits = tf.squeeze(logits)
        labels_normlized = tf.cast(labels, dtype=tf.float32)
        if loss_weight is not None:
            class_weights = tf.gather(loss_weight, labels)
            losses = tf.losses.mean_squared_error(labels_normlized, logits, weights=class_weights)
        else:
            losses = tf.losses.mean_squared_error(labels_normlized, logits)
        loss = tf.reduce_mean(losses)
    return loss


def sparse_softmax_focal_loss(logits, labels, gamma=2.0, alpha=0.25):
    num_cls = logits.shape[1]
    onehot_labels = tf.one_hot(labels, num_cls)
    logits = tf.clip_by_value(tf.nn.softmax(logits), 1e-10, 1.)
    cross_entropy = tf.multiply(onehot_labels, -tf.log(logits))
    weight = tf.multiply(onehot_labels, tf.pow(tf.subtract(1., logits), gamma))
    loss = tf.reduce_mean(tf.multiply(alpha, tf.multiply(weight, cross_entropy)))
    return loss


def sigmoid_focal_loss(logits, labels, gamma=2.0, alpha=0.25, epsilon=1e-10):
    with tf.name_scope("sigmoid_focal_loss"):
        labels = tf.cast(labels, tf.float32)
        logits = tf.nn.sigmoid(logits)
        logits = tf.where(tf.equal(labels, 1.), logits, tf.subtract(1., logits))
        weight = tf.multiply(tf.negative(alpha), tf.pow(tf.subtract(1., logits), gamma))
        losses = tf.reduce_sum(tf.multiply(weight, tf.log(logits + epsilon)), axis=1)
        loss = tf.reduce_mean(losses)
    return loss


def loss_multi_label(logits, labels, loss_weights_single=None):
    with tf.name_scope("multi_label_loss"):
        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=logits)
        # for weight
        if loss_weights_single is not None:
            losses = tf.multiply(losses, loss_weights_single)
        losses = tf.reduce_sum(losses, axis=1)
        losses = tf.reduce_mean(losses)
    return losses


def loss_double_multi_label(logits, labels, num_class, loss_weights_single=None, loss_weights_double=None):
    """

    :param logits: batch_size * (num_class * 2)
    :param labels: batch_size * (num_class * 2)
    :return:
    """
    with tf.name_scope("double_multi_label_loss"):
        labels = tf.concat([labels, 1 - labels], axis=-1)
        batch_size = tf.shape(logits)[0]
        logits_double = tf.reshape(logits, shape=[batch_size, 2, num_class])
        logits_double_unstacked = tf.unstack(logits_double, axis=-1)
        labels_double = tf.reshape(labels, shape=[batch_size, 2, num_class])
        labels_double_unstacked = tf.unstack(labels_double, axis=-1)
        losses = []
        for i in range(num_class):
            singel_label = tf.argmax(labels_double_unstacked[i], axis=-1)
            if loss_weights_double is not None:
                binary_loss = loss_single_label(logits_double_unstacked[i], singel_label, loss_weight=loss_weights_double[i])
            else:
                binary_loss = loss_single_label(logits_double_unstacked[i], singel_label)
                # binary_loss = focal_loss(logits_double_unstacked[i], singel_label)

            losses.append(binary_loss)
        if loss_weights_single is not None:
            loss = tf.reduce_sum(tf.multiply(tf.convert_to_tensor(losses), loss_weights_single))
        else:
            loss = tf.add_n(losses)
    return loss


def l2_loss(l2_lambda=0.0001):
    with tf.variable_scope('l2_loss'):
        l2_loss = tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        return l2_loss


def list_to_multi_hot(labels, num_class):
    multi_hot = np.zeros((len(labels), num_class))
    for i in range(len(labels)):
        removed_label_index = np.argwhere(labels[i] == -1)
        real_labels = np.delete(labels[i], removed_label_index)
        multi_hot[i][real_labels] = 1.
    return multi_hot


def list_to_double_multi_hot(labels, num_class):
    multi_hot = list_to_multi_hot(labels, num_class)
    res_multi_hot = 1.0 - multi_hot
    double_multi_hot = np.concatenate([multi_hot, res_multi_hot], axis=1)
    return double_multi_hot

def id_to_value(labels, id2value):
    # label_list = [id2value.get(label) for label in labels.tolist()]
    label_list = []
    for label in labels.tolist():
        if label == -2:
            label_list.append(500)
        elif label == -1:
            label_list.append(400)
        else:
            label_list.append(id2value.get(label))
    return np.array(label_list)

def region_embedding(inputs, channel_size, filter_size, embed_size, is_training=True):
    """
    region embedding from textcnn
    :param inputs: words embedding
    :param channel_size:
    :param filter_size:
    :param embed_size:
    :param is_training:
    :return:
    """
    inputs = tf.pad(inputs, paddings=tf.constant([[0, 0], [1, 1], [0, 0]]))
    inputs = tf.expand_dims(inputs, -1)
    with tf.name_scope("region_embedding"):
        with slim.arg_scope([slim.conv2d], padding='VALID', normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training}):
            net = slim.conv2d(inputs,
                              channel_size,
                              [filter_size, embed_size],
                              scope='conv_{}'.format(filter_size))
            return net


def conv_block(inputs, name, channel_size, max_pooling=True, is_training=True):
    if max_pooling:
        with tf.variable_scope("{}_max_pooling".format(name)):
            with slim.arg_scope([slim.max_pool2d], kernel_size=[3, 1], stride=2, padding='SAME'):
                inputs = slim.max_pool2d(inputs)
    shortcut = inputs
    for i in range(2):
        with tf.variable_scope("{}_conv_{}".format(name, i)):
            with slim.arg_scope([slim.conv2d],
                                padding='SAME',
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': is_training,
                                                   "scale": True,
                                                   "epsilon": 1e-5}):
                inputs = slim.conv2d(inputs, channel_size, [3, inputs.get_shape()[2]])

    inputs = inputs + shortcut
    return inputs


def bi_lstm(inputs, level, hidden_size, keep_prob, reuse=False):
    """

    :param inputs:
    :param level:
    :param hidden_size:
    :param keep_prob:
    :param reuse:
    :return:
    """
    # batch_size = inputs.get_shape().as_list()[0]
    with tf.variable_scope('bi_lstm_{}'.format(level), reuse=reuse):
        fw_cell = rnn.GRUCell(hidden_size)
        fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=keep_prob)
        bw_cell = rnn.GRUCell(hidden_size)
        bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=keep_prob)
        # init_fw_state = tf.get_variable("init_fw_stata", shape=[batch_size, hidden_size])
        # init_bw_state = tf.get_variable("init_bw_stata", shape=[batch_size, hidden_size])
        outputs, hidden_states = tf.nn.bidirectional_dynamic_rnn(fw_cell,
                                                                 bw_cell,
                                                                 inputs,
                                                                 dtype=tf.float32)
                                                                 # initial_state_fw=init_fw_state,
                                                                 # initial_state_bw=init_bw_state)
    outputs = tf.concat(outputs, axis=2)
    hidden_states = tf.concat([hidden_states[0], hidden_states[1]], axis=1)
    return outputs, hidden_states


def project_output(inputs, name, num_class, keep_prob, is_training=True):
    with tf.variable_scope("output_{}".format(name)):
        with slim.arg_scope([slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training}):
            net = slim.fully_connected(inputs, num_outputs=1024, scope='liner_{}_0'.format(name))
            net = slim.dropout(net, keep_prob=keep_prob)
        with slim.arg_scope([slim.fully_connected], activation_fn=None):
            logits = slim.fully_connected(net, num_outputs=num_class, scope=name)
            return logits


def attention_multihop(inputs, level, num_hops, reuse=False,
                       initializer=tf.random_normal_initializer(stddev=0.1)):
    """

    :param inputs:
    :param level:
    :param num_hops:
    :param reuse:
    :param initializer:
    :return:
    """
    num_units = inputs.get_shape().as_list()[-1] / num_hops  # get last dimension
    attention_rep_list=[]
    for i in range(num_hops):
        with tf.variable_scope("attention_{}_{}".format(i, level), reuse=reuse):
            v_attention = tf.get_variable("u_attention_{}".format(level),
                                          shape=[num_units],
                                          initializer=initializer)
            # 1.one-layer MLP
            u = tf.layers.dense(inputs, num_units, activation=tf.nn.tanh, use_bias=True)
            # 2.compute weight by compute simility of u and attention vector v
            score = tf.multiply(u, v_attention)  # [batch_size,seq_length,num_units]
            weight = tf.reduce_sum(score, axis=2, keep_dims=True)  # [batch_size,seq_length,1]
            # 3.weight sum
            attention_rep = tf.reduce_sum(tf.multiply(u, weight), axis=1)
            attention_rep_list.append(attention_rep)

    attention_representation = tf.concat(attention_rep_list, axis=-1)
    return attention_representation


def diff_loss(shared_feat, task_feat):
    """
    Orthogonality Constraints
    :param shared_feat:
    :param task_feat:
    :return:
    """
    task_feat -= tf.reduce_mean(task_feat, 0)
    shared_feat -= tf.reduce_mean(shared_feat, 0)
    task_feat = tf.nn.l2_normalize(task_feat, 1)
    shared_feat = tf.nn.l2_normalize(shared_feat, 1)

    correlation_matrix = tf.matmul(task_feat, shared_feat, transpose_a=True)
    cost = tf.reduce_mean(tf.square(correlation_matrix))
    cost = tf.where(cost > 0, cost, 0, name='value')

    assert_op = tf.Assert(tf.is_finite(cost), [cost])
    with tf.control_dependencies([assert_op]):
        loss_diff = tf.identity(cost)
    return loss_diff


def swa_model_ensemble(sess, model_ckpt_path, swa_epoch):
    if swa_epoch == 0 and tf.gfile.Exists(os.path.join(model_ckpt_path, "model.best.index")):
        os.system("cp {} {}".format(os.path.join(model_ckpt_path, "model.best.data-00000-of-00001"),
                                    os.path.join(model_ckpt_path, 'model.swa_{}.data-00000-of-00001'.format(swa_epoch))))
        os.system("cp {} {}".format(os.path.join(model_ckpt_path, "model.best.index"),
                                    os.path.join(model_ckpt_path,
                                                 'model.swa_{}.index'.format(swa_epoch))))
        os.system("cp {} {}".format(os.path.join(model_ckpt_path, "model.best.meta"),
                                    os.path.join(model_ckpt_path,
                                                 'model.swa_{}.meta'.format(swa_epoch))))
    else:
        var_list = slim.get_variables_to_restore(exclude=['global_step'])
        var_values, var_dtypes = {}, {}
        for name, shape in var_list:
            var_values[name] = np.zeros(shape)

        best_reader = load_checkpoint(os.path.join(model_ckpt_path, "model.best"))
        last_swa_reader = load_checkpoint(os.path.join(model_ckpt_path, "model.swa_{}".format(swa_epoch - 1)))
        for name in var_list:
            best_tensor = best_reader.get_tensor(name)
            last_swa_tensor = last_swa_reader.get_tensor(name)
            var_dtypes[name] = best_tensor.dtype
            var_values[name] = (last_swa_tensor * swa_epoch + best_tensor) / (swa_epoch + 1)

        tf_vars = [tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[v]) for v in var_values]
        placeholds = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
        assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholds)]
        global_step = tf.Variable(0, name="global_step", trainable=False,dtype=tf.int64)
        saver = tf.train.Saver(tf.all_variables)
        sess.run(tf.initialize_all_variables())
        for p, assign_ops, (name, value) in zip(placeholds, assign_ops, six.iteritems(var_values)):
            sess.run(assign_ops, {p: value})
        saver.save(sess, os.path.join(model_ckpt_path, 'model.swa_{}'.format(swa_epoch)), write_state=False)


if __name__ == "__main__":
    def alter_data(_data):
        data = _data.copy()
        new_data = []
        for d in data:
            for i, l in enumerate(d):
                if np.random.rand() < 0.2:
                    d[i] = (d[i] + 1) % 2
            new_data.append(d)
        return np.concatenate([np.array(new_data), 1 - np.array(new_data)], axis=1)


    def get_data():
        # Number of different classes
        num_classes = 10
        classes = list(range(num_classes))
        # Numberof samples in synthetic dataset
        examples = 10000
        # Max number of labels per sample. Minimum is 1
        max_labels = 5
        class_probabilities = np.array(
            list(6 * np.exp(-i * 5 / num_classes) + 1 for i in range(num_classes))
        )
        class_probabilities /= class_probabilities.sum()
        labels = [
            np.random.choice(
                classes,  # Choose labels in 0..num_classes
                size=np.random.randint(1, max_labels),  # number of labels for this sample
                p=class_probabilities,  # Probability of drawing each class
                replace=False,  # A class can only occure once
            )
            for _ in range(examples)  # Do it `examples` times
        ]
        y_true = np.zeros((examples, num_classes)).astype(np.int64)
        for i, l in enumerate(labels):
            y_true[i][l] = 1
        y_pred = alter_data(y_true)
        return y_true, y_pred


    np.random.seed(0)

    y_true, y_pred = get_data()
    num_classes = y_true.shape[-1]

    with tf.Graph().as_default():
        t = tf.placeholder(tf.int64, [None, None], "y_true")
        p = tf.placeholder(tf.int64, [None, None], "y_pred")

        counts, update = streaming_counts(t, p, num_classes)
        f1 = streaming_f1(counts)
        tf_f1 = cal_double_multi_label_score(p, t, num_classes)

        with tf.Session() as sess:
            tf.local_variables_initializer().run()
            mic, mac, wei = sess.run(tf_f1, feed_dict={t: y_true, p: y_pred})
            print('\n', mic, mac, wei)
            # for i in range(len(y_true) // 100):
            #     y_t = y_true[i * 100: (i + 1) * 100].astype(np.int64)
            #     y_p = y_pred[i * 100: (i + 1) * 100].astype(np.int64)
            #     _ = sess.run([update], feed_dict={t: y_t, p: y_p})
            #     # print('\n', tf_f1_)
            # mic, mac, wei = [f.eval() for f in f1]
            # print("\n", mic, mac, wei)