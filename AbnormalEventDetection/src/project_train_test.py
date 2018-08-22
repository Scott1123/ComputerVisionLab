# coding=utf-8
import cv2
import numpy as np
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from fnmatch import fnmatch


def main():
    S = []
    project_path = os.path.dirname(os.path.realpath(__file__))
    if os.path.exists(project_path + r'\check_point_model\check_point_S'):
        S = pickle.load(open(project_path + r"\check_point_model\check_point_S", 'rb'))
    else:
        train_frames_path = project_path + r"\all_train_frames"
        S = preprocess_and_training(train_frames_path)
        pickle.dump(S, open(project_path + r'\check_point_model\check_point_S', 'wb'))
    test_frames_path = project_path + r'\all_test_frames'
    import_and_test_abnormal(test_frames_path, S)


def import_and_test_abnormal(test_frames_path, S):
    test_folder_list = []
    for path, subdirs, files in os.walk(test_frames_path):
        for name in subdirs:
            test_folder_list.append(str(os.path.join(path, name)))
    feature_list = []
    for folder in test_folder_list:
        feature_list = framesToFeatures(folder, "*.jpg")
        file_list, result = testing_algorithm(feature_list, S, 0.00001915625)
        if (len(result) == 0):
            print "Normal"
        else:
            for res in result:
                print res
            continue_key = raw_input("Press enter to show the abnormal frames : ")
            key_str = "1"
            while (key_str == "1"):
                if (continue_key == ""):
                    show_image(folder, file_list)
                    key_str = raw_input("Press Enter to continue or 1 to replay : ")


def preprocess_and_training(train_frames_path):
    i = 0
    file_list1 = []
    for path, subdirs, files in os.walk(train_frames_path):
        for name in subdirs:
            file_list1.append(str(os.path.join(path, name)))
    feature_list = []

    for folder in file_list1:
        feature_list.append(framesToFeatures(folder, "*.jpg"))
        i += 1

    # Training for Sparse Combination Learning
    S = []
    B = []

    feature_list = [i[:100] for i in feature_list]
    # feature_list = [i for i in feature_list]
    for set in feature_list:
        S_temp, B_temp = training_algorithm(set)
        S += S_temp
        B += B_temp
    return S


def training_algorithm(X):
    Xc = X
    S = []
    B = []
    gamma = []
    i = 1
    while (len(Xc) > 10):
        # Create the initial dictionary Si using kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(np.array(Xc, dtype="float32"), 10, None, criteria, 10, flags)
        centers = [(sum(val) / len(val)) for val in centers]
        Si = [centers]
        # Reset Gamma and Beta i for next Vector generation
        gamma = []
        Bi = []
        epoch = 0
        start = 1

        while (start == 1 or start == 2 or deltaL < 0):

            if (start == 2):
                start = 0
            if (start == 1):
                deltaL = 0
                start = 2
                L2 = 0
                L1 = 0
            Bi = optimise_beta(Si, Xc)
            Si = np.subtract(np.array(Si), (0.0001 * deltaL))
            gamma = optimise_gamma(Si, Xc, Bi, 0.04)
            L1 = L2
            L2 = evaluate_L(Si, Xc, Bi, gamma)
            deltaL = L2 - L1
            epoch += 1
        S.append(Si)
        B.append(Bi)
        change_index = 0
        for val in range(len(gamma)):
            if (gamma[val] == 0):
                del Xc[val - change_index]
                change_index += 1
        i += 1
    return S, B


def optimise_beta(Si, Xc):
    # Using equation 6 optimise beta value

    beta = []
    Si = np.array(Si)
    Si_transpose = np.transpose(Si)
    m = 0.00000003

    # print Si.shape
    # print Si_transpose.shape

    for xj in Xc:
        numpy_xj = np.array([xj])
        Si_T_Si = np.dot(Si_transpose, Si)

        # TODO: funcion *det* may cause errors
        # [[4, 2], [10, 5]]
        if (np.linalg.det(Si_T_Si) == 0):
            Si_T_Si = np.add(Si_T_Si, m * np.eye(10, 10))

        inverse_sit = np.linalg.inv(Si_T_Si)
        dot_in_si = np.dot(inverse_sit, Si_transpose)
        itr_beta = np.dot(dot_in_si, numpy_xj)
        beta.append(itr_beta)

    return beta


def optimise_gamma(Si, Xc, Bi, lamda):
    gamma = []
    Si = np.array(Si)
    for xj in range(len(Xc)):
        if ((((np.linalg.norm(np.subtract(np.array([Xc[xj]]), np.dot(Si, np.array(Bi[xj]))))) ** 2) ** 2) < lamda):
            gamma.append(1)
        else:
            gamma.append(0)
    return gamma


def evaluate_L(Si, Xc, Bi, gamma):
    L = 0
    Si = np.array(Si)
    temp_l = []

    for xj in range(len(Xc)):
        l_iter_val = gamma[xj] * (
                    ((np.linalg.norm(np.subtract(np.array([Xc[xj]]), np.dot(Si, np.array(Bi[xj]))))) ** 2) ** 2)
        temp_l.append(l_iter_val)

    return sum(temp_l)


def framesToFeatures(frames_path, pattern="*.jpg"):
    print(frames_path)
    features = []
    file_list = []
    for path, subdirs, files in os.walk(frames_path):
        for name in files:
            if fnmatch(name, pattern):
                file_list.append(str(os.path.join(path, name)))
    numOfFiles = len(file_list)

    i = 0
    time = 0
    number_cubes = 0
    # reading 5 images at a time
    while (numOfFiles - i >= 5):
        time += 1
        # 转换为灰度图
        img1 = cv2.cvtColor(cv2.imread(file_list[i]), cv2.COLOR_BGR2GRAY);
        i += 1;
        img2 = cv2.cvtColor(cv2.imread(file_list[i]), cv2.COLOR_BGR2GRAY);
        i += 1;
        img3 = cv2.cvtColor(cv2.imread(file_list[i]), cv2.COLOR_BGR2GRAY);
        i += 1;
        img4 = cv2.cvtColor(cv2.imread(file_list[i]), cv2.COLOR_BGR2GRAY);
        i += 1;
        img5 = cv2.cvtColor(cv2.imread(file_list[i]), cv2.COLOR_BGR2GRAY);
        i += 1;

        image_set = [img1, img2, img3, img4, img5]

        # Create 3 different scale for each image

        re_img_2020_set = []
        re_img_4030_set = []
        re_img_160120_set = []
        for image in image_set:
            img_2020 = cv2.resize(image, (20, 20))
            img_4030 = cv2.resize(image, (40, 30))
            img_160120 = cv2.resize(image, (160, 120))

            re_img_2020_set.append(img_2020)
            re_img_4030_set.append(img_4030)
            re_img_160120_set.append(img_160120)

        resize_image_set = [re_img_2020_set, re_img_4030_set, re_img_160120_set]

        # Collect non-overlaping patches form all the scale

        patches_all = [[], [], []]
        i1 = 0

        for images_set in resize_image_set:
            for resize_img in images_set:
                patch_list = []
                patch = []
                for start in range(0, len(resize_img[0]), 10):
                    count = 1
                    for row in resize_img:
                        patch.append(row[start:start + 10])

                        if (count == 10):
                            count = 0
                            patch_list.append(patch)
                            patch = []
                        count += 1
                patches_all[i1].append(patch_list)
            i1 += 1

        # Generate cubes and list of all cubes

        cubes = []

        for resolution_patch_set in patches_all:
            for i1 in range(len(resolution_patch_set[0])):
                p_one = resolution_patch_set[0][i1];
                p_two = resolution_patch_set[1][i1];
                p_three = resolution_patch_set[2][i1];
                p_four = resolution_patch_set[3][i1];
                p_five = resolution_patch_set[4][i1];
                cubes.append([p_one, p_two, p_three, p_four, p_five])

        number_cubes += len(cubes)

        # features=[]
        for cub in cubes:
            # 表示的是示导的阶数，0 表示这个方向上没有求导，一般为 0，1，2
            # Calculate the x, y and t derivative for each cubes
            sobelx = cv2.Sobel(np.array(cub), cv2.CV_64F, 1, 0, ksize=-1)
            sobely = cv2.Sobel(np.array(cub), cv2.CV_64F, 0, 1, ksize=-1)
            sobelt = cv2.Sobel(np.array(zip(*cub)), cv2.CV_64F, 0, 1, ksize=-1)
            sobelt = zip(*sobelt)

            feature = []
            # feature=np.array(feature)

            # Concatinate all the x,y,t values at each pixel to generate 1500 dimension feature

            for time_value in range(5):
                for y_value in range(10):
                    for x_value in range(10):
                        feature.append(sobelx[time_value][y_value][x_value])
                        feature.append(sobely[time_value][y_value][x_value])
                        feature.append(sobelt[time_value][y_value][x_value])
            features.append(feature)

    print "--------------Done Feature Extraction------------"
    print "Number of cubes generated : ", number_cubes
    print "Number of feature generated : ", len(features)
    print "Length of each feature : ", len(features[0])
    print "-------------------------------------------------"
    return features


def show_image(folder, file_list_no, pattern="*.png"):
    file_list = []

    for path, subdirs, files in os.walk(folder):
        for name in files:
            # if fnmatch(name, pattern):
            file_list.append(str(os.path.join(path, name)))

    numOfFiles = len(file_list)
    # print file_list
    file_to_print = []

    for f_no in file_list_no:
        if (f_no == 0):
            file_to_print.append(file_list[0])
            file_to_print.append(file_list[1])
            file_to_print.append(file_list[2])
            file_to_print.append(file_list[3])
            file_to_print.append(file_list[4])
            file_to_print.append(file_list[5])
        elif (f_no <= numOfFiles):
            file_to_print.append(file_list[f_no - 1])
            if (f_no - 1 + 1 < numOfFiles):
                file_to_print.append(file_list[f_no - 1 + 1])
            if (f_no - 1 + 2 < numOfFiles):
                file_to_print.append(file_list[f_no - 1 + 2])
            if (f_no - 1 + 3 < numOfFiles):
                file_to_print.append(file_list[f_no - 1 + 3])
            if (f_no - 1 + 4 < numOfFiles):
                file_to_print.append(file_list[f_no - 1 + 4])
            if (f_no - 1 + 5 < numOfFiles):
                file_to_print.append(file_list[f_no - 1 + 5])
        else:
            break

    for file in file_to_print:
        img = cv2.imread(file)  # read a picture using OpenCV
        cv2.imshow('image', img)  # Display the picture
        cv2.waitKey(150)  # wait for closing
    cv2.destroyAllWindows()


def testing_algorithm(x, S, T):
    # print S
    R = getR(S)
    # print R
    return_list = []
    file_list = []
    # print xs
    i = 0
    time = 0
    flag = 0
    for xi in x:
        i += 1
        flag = 0
        mean = []
        for Ri in R:
            val = np.linalg.norm(np.dot(np.array(Ri), np.array([xi]))) ** 2

            mean.append(val)

            if (val < T):
                flag = 1
                break
        if (i == 208):
            i = 0
            min_mean = min(mean)
            if ((str("Abnormal at time" + str(time) + " seconds.") not in return_list) and min_mean > 0.0000000014):
                return_list.append(str("Abnormal at time" + str(time) + " seconds."))
                file_list.append(time)
            print "time:small : ", time, min_mean
            time += 5
            mean = []

    return file_list, return_list


def getR(S):
    R = [];
    m = 0.00000003

    for Si in S:
        Si = np.array(Si);
        Si_transpose = np.transpose(Si);
        Si_T_Si = np.dot(Si_transpose, Si)
        if (np.linalg.det(Si_T_Si) == 0):
            Si_T_Si = np.add(Si_T_Si, m * np.eye(10, 10))

        Ri = np.subtract(np.dot(Si, np.dot(np.linalg.inv(Si_T_Si), Si_transpose)), np.identity(len(Si)));
        R.append(Ri);

    return R;


main()


def search_threhold(candidate_t):
    pass