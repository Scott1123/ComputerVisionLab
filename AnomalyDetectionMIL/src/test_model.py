from datetime import datetime
from keras.models import load_model

from utils import *

seed = 7
np.random.seed(seed)

print("Starting testing...")

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

All_Test_files = os.listdir(TEST_DATA_DIR)
All_Test_files.sort()

model = load_model(MODEL_PATH)
nVideos = len(All_Test_files)
time_before = datetime.now()

for iv in range(nVideos):
    Test_Video_Path = os.path.join(TEST_DATA_DIR, All_Test_files[iv])
    inputs = load_test_data_one_video(Test_Video_Path)  # 32 segments features for one testing video
    predictions = model.predict_on_batch(inputs)  # Get anomaly prediction for each of 32 video segments.
    aa = All_Test_files[iv]
    aa = aa[0:-4]
    A_predictions_path = RESULTS_DIR + str(aa) + '.mat'
    # Save array of 1*32, containing anomaly score for each segment.
    # Please see Evaluate Anomaly Detector to compute ROC.

    print("Total Time took: " + str(datetime.now() - time_before))
