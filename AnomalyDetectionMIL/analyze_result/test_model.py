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


def test(test_dir, res_dir, ground_truth, model_path):
    if os.path.exists(res_dir):
        shutil.rmtree(res_dir)
    os.makedirs(res_dir)
    
    all_test_file = os.listdir(test_dir)
    all_test_file.sort()
    
    model = load_model(model_path, custom_objects={'custom_loss': custom_loss})
    num_videos = len(all_test_file)
    time_start = datetime.now()

    for file in all_test_file:
        path = os.path.join(test_dir, file)
        inputs = load_test_data_one_video(path)
        predictions = model.predict_on_batch(inputs)
        label = 1 if np.max(predictions) >= threshold else 0
        save_path = os.path.join(res_dir, file)
        with open(save_path, 'w') as f:
            f.write('%d %d' % (label, ground_truth))
            for item in predictions:
                f.write(str(item[0]))
                f.write('\n')


if __name__ == '__main__':
    test(TEST_ABNORMAL, RESULT_ABNORMAL, 0, MODEL_FOR_TEST)
    print('test abnormal... OK')
    test(TEST_NORMAL, RESULT_NORMAL, 1, MODEL_FOR_TEST)
    print('test normal... OK')
