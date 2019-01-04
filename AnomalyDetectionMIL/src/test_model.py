from __future__ import division
from datetime import datetime
from keras.models import load_model
from scipy.io import savemat
import shutil
from utils import *
import tensorflow as tf

tf.python.control_flow_ops = tf
# import keras.losses
# keras.losses.custom_loss = custom_loss


seed = 0
np.random.seed(seed)

print("starting testing...")

model = load_model(MODEL_FOR_TEST, custom_objects={'custom_loss': custom_loss})
print('loading model... OK')

times = 10 ** 13


def test(test_dir, res_dir, ground_truth):
    if os.path.exists(res_dir):
        shutil.rmtree(res_dir)
    os.makedirs(res_dir)

    all_test_file = os.listdir(test_dir)
    all_test_file.sort()

    num_videos = len(all_test_file)
    time_start = datetime.now()

    score_list = []
    acc = 0
    for file in all_test_file:
        path = os.path.join(test_dir, file)
        inputs = load_test_data_one_video(path)
        predictions = model.predict_on_batch(inputs)
        predictions = predictions * times
        max_element = np.max(predictions)
        score_list.append(max_element)
        label = (1 if max_element >= threshold else 0)
        if label == ground_truth:
            acc += 1
        print(file + ':')
        print('label:%d  ground_truth:%d' % (label, ground_truth))
        pred = sum(predictions.tolist(), [])
        print(pred)
        print('')
        save_path = os.path.join(res_dir, file)
        with open(save_path, 'w') as f:
            f.write('%d %d\n' % (label, ground_truth))
            for item in predictions:
                f.write(str(item[0]))
                f.write('\n')
    return acc / num_videos, score_list


if __name__ == '__main__':
    abnormal_acc, abnormal_score = test(TEST_ABNORMAL, RESULT_ABNORMAL, 0)
    print('test abnormal... OK')
    normal_acc, normal_score = test(TEST_NORMAL, RESULT_NORMAL, 1)
    print('test normal... OK')
    print(abnormal_acc, normal_acc)
    print('end.')
    abnormal_score_path = os.path.join(OUTPUT_DIR, 'abnormal_score_list.txt')
    normal_score_path = os.path.join(OUTPUT_DIR, 'normal_score_list.txt')
    with open(abnormal_score_path, 'w') as f:
        for score in abnormal_score:
            f.write(str(score) + '\n')
    with open(normal_score_path, 'w') as f:
        for score in normal_score:
            f.write(str(score) + '\n')

