import datetime
from os import listdir

from keras.regularizers import l2
from keras.models import Sequential
from keras.optimizers import Adagrad
from keras.layers import Dense, Dropout

from AnomalyDetectionMIL.TransCan.utils import *

print("1. Create model...")

model = Sequential()
model.add(Dense(512, input_dim=4096, init='glorot_normal', W_regularizer=l2(0.001), activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(32, init='glorot_normal', W_regularizer=l2(0.001)))
model.add(Dropout(0.6))
model.add(Dense(1, init='glorot_normal', W_regularizer=l2(0.001), activation='sigmoid'))

adagrad = Adagrad(lr=0.01, epsilon=1e-08)

model.compile(loss=custom_objective, optimizer=adagrad)

print("2. Starting training...")

AllClassPath = '/data/UCF_Anomaly_Dataset/C3D_Features/Train/'
output_dir = '/data/UCF_Anomaly_Dataset/Trained_Models/TrainedModel_MIL_C3D/'
weights_path = output_dir + 'weights.mat'
model_path = output_dir + 'model.json'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

All_class_files = listdir(AllClassPath)
All_class_files.sort()
loss_graph = []
num_iters = 8000  # 20000
total_iterations = 0
batchsize = 60
time_before = datetime.now()

for it_num in range(num_iters):

    AbnormalPath = os.path.join(AllClassPath, All_class_files[0])  # Path of abnormal already computed C3D features
    NormalPath = os.path.join(AllClassPath, All_class_files[1])  # Path of Normal already computed C3D features
    inputs, targets = load_dataset_Train_batch(AbnormalPath, NormalPath)  # Load normal and abnormal video C3D features
    batch_loss = model.train_on_batch(inputs, targets)
    loss_graph = np.hstack((loss_graph, batch_loss))
    total_iterations += 1
    if total_iterations % 20 == 1:
        print("These iteration=" +
              str(total_iterations) + ") took: " +
              str(datetime.now() - time_before) +
              ", with loss of " + str(batch_loss))
        iteration_path = output_dir + 'Iterations_graph_' + str(total_iterations) + '.mat'
        savemat(iteration_path, dict(loss_graph=loss_graph))
    if total_iterations % 1000 == 0:  # Save the model at every 1000th iterations.
        weights_path = output_dir + 'weightsAnomalyL1L2_' + str(total_iterations) + '.mat'
        save_model(model, model_path, weights_path)

save_model(model, model_path, weights_path)
