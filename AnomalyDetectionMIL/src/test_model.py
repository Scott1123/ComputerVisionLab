from datetime import datetime
from keras.models import load_model
from scipy.io import savemat

from utils import *

seed = 7
np.random.seed(seed)

print("Starting testing...")

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

all_test_file = os.listdir(TEST_DATA_DIR)
all_test_file.sort()

model = load_model(MODEL_PATH)
num_videos = len(all_test_file)
time_start = datetime.now()

for i in range(num_videos):
    test_video_path = os.path.join(TEST_DATA_DIR, all_test_file[i])
    inputs = load_test_data_one_video(test_video_path)  # 32 segments features for one testing video
    predictions = model.predict_on_batch(inputs)  # Get anomaly prediction for each of 32 video segments.
    name = all_test_file[i]
    name = name[:-4]  # remove suffix
    predictions_mat = RESULTS_DIR + str(name) + '_pred.mat'
    savemat(predictions_mat)

    print("Total Time took: " + str(datetime.now() - time_start))
