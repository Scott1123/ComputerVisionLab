import numpy as np


TRAIN_DATA_DIR = 'E:/github_projects/data/UCF_Anomaly_Dataset/C3D_Features/Train/'

a = np.random.randint(1, 100, (5, 10))

a = a / 1000
print(a)

with open(TRAIN_DATA_DIR + '1.txt', 'w') as f:
    for i in range(5):
        for j in range(10):
            f.write('%.4f ' % (a[i][j]))
        f.write('\n')

print('end.')
