import tensorflow as tf
import keras.backend as K
import numpy as np
# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()
K.set_floatx('float32')

# parameters
batch_size = 6  # train batch
one_video_seg = 5  # single video
one_video_feat = 5  # single video
one_batch_feat = one_video_seg * batch_size  # for one batch

num_abnormal = 6
num_normal = 6

# hyper_parameter
lambda_1 = 0.0008  # for temporal_smoothness_term
lambda_2 = 0.0008  # for sparsity_term


data_type = tf.float64


def custom_loss(y_true, y_pred):
    # init
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    # terms
    sum_true = K.variable([], dtype=data_type)
    sum_pred = K.variable([], dtype=data_type)  # sparsity term
    max_pred = K.variable([], dtype=data_type)
    pow_dif_pred = K.variable([], dtype=data_type)  # temporal smoothness term

    for i in range(batch_size):
        # init sum_true
        tmp_true = y_true[i * one_video_seg: i * one_video_seg + one_video_seg]
        sum_true = K.concatenate([sum_true, [K.sum(tmp_true, axis=-1)]])

        # init sum_pred, max_pred
        tmp_pred = y_pred[i * one_video_seg: i * one_video_seg + one_video_seg]
        sum_pred = K.concatenate([sum_pred, [K.sum(tmp_pred, axis=-1)]])
        max_pred = K.concatenate([max_pred, [K.max(tmp_pred, axis=-1)]])

        # calculate temporal_smoothness_term
        # dif = [tmp_pred[k] - tmp_pred[k + 1] for k in range(one_video_seg - 1)]
        # print(type(tmp_pred))
        one = K.ones_like(tmp_pred)
        v0 = K.concatenate([one, tmp_pred])
        v1 = K.concatenate([tmp_pred, one])
        dif = (v1[:one_video_seg+1] - v0[one_video_seg-1:])[1:]
        dif = K.concatenate([dif, [tmp_pred[one_video_seg - 1] - 1]])
        pow_dif_pred = K.concatenate([pow_dif_pred, [K.sum(K.pow(dif, 2))]])

    preds = max_pred
    trues = sum_true

    sparsity_term = K.sum(sum_pred, axis=-1)  # 0:batch_size//2 ?
    temporal_smoothness_term = K.sum(pow_dif_pred, axis=-1)  # 0:batch_size//2 ?

    # get normal & abnormal preds
    normal_pred = tf.boolean_mask(preds, K.equal(trues, one_video_seg))
    abnormal_pred = tf.boolean_mask(preds, K.equal(trues, 0))

    loss = K.variable([], dtype=data_type)
    for i in range(batch_size // 2):
        p0 = K.maximum(K.cast(0, dtype=data_type), 1 - abnormal_pred[i] + normal_pred[i])
        # print(i, K.eval(p0))
        loss = K.concatenate([loss, [p0]])

    loss = tf.reduce_mean(loss) + lambda_1 * temporal_smoothness_term + lambda_2 * sparsity_term

    return loss


t = np.zeros((batch_size, one_video_seg), dtype="float64")
t[2] = 1
t[3] = 1
t[5] = 1
p = np.zeros((batch_size, one_video_seg), dtype="float64")
p[2] = 1
p[5] = 1
t = K.variable(t, dtype="float64")
p = K.variable(p, dtype="float64")

loss = custom_loss(t, p)

print(loss)
print(K.eval(loss))
