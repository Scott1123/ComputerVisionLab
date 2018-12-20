import numpy as np


TRAIN_DATA_DIR = 'E:/github_projects/data/UCF_Anomaly_Dataset/C3D_Features/Train/'

a = np.random.randint(1, 100, (32, 1))
a = a.tolist()
b = sum(a, [])

print(b)
