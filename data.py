import pandas as pd
import numpy as np
import os
import time
import glob
from keras.utils import np_utils
from tqdm import tqdm
import math
import cv2
import random

class Data:
    def __init__(self) -> None:
        self.normalization = None
        self.img_rows, self.img_cols = None, None
        random_state = 51
        use_cache = 1
        color_type_global = 3
        imgs_list_path = "state-farm-distracted-driver-detection/driver_imgs_list.csv"
        driver_imgs_list = pd.read_csv(imgs_list_path)
        driver_imgs_list.head()

    def get_im_cv2(self,path, color_type=3):
        
        # Load as grayscale
        if color_type == 1:
            img = cv2.imread(path, 0)
        elif color_type == 3:
            img = cv2.imread(path)

        resized = cv2.resize(img, (self.img_cols, self.img_rows))

        return resized

    def get_im_cv2_mod(self,path, color_type=3):
        # Load as grayscale
        #print("READ PATH",path)
        #print("_____________________________")
        if color_type == 1:
            img = cv2.imread(path, 0)
        else:
            img = cv2.imread(path)
        # Reduce size
        
        rotate  = random.uniform(-10, 10)
        M       = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), rotate, 1)
        img     = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        resized = cv2.resize(img, (self.img_cols, self.img_rows), cv2.INTER_LINEAR)
        
        return resized

    def get_driver_data(self):
        dr = dict()

        path = "state-farm-distracted-driver-detection/driver_imgs_list.csv"
        #path = os.path.join('driver_imgs_list.csv')
        print('Read drivers data')
        f = open(path, 'r')
        line = f.readline()
        while (1):
            line = f.readline()
            if line == '':
                break
            arr = line.strip().split(',')
            dr[arr[2]] = arr[0]
        f.close()
        return dr


    def load_train(self, color_type=3):
        X_train = []
        y_train = []
        driver_id = []
        start_time = time.time()
        driver_data = self.get_driver_data()

        for j in range(10):
            
            print('Load folder c{}'.format(j))
            path = os.path.join('state-farm-distracted-driver-detection','imgs', 'train', 'c' + str(j), '*.jpg')
            files = glob.glob(path)
            

            k = 0
            for fl in tqdm(files):
                if k>100:
                    break
                flbase = os.path.basename(fl)
                img = self.get_im_cv2_mod(fl, color_type)
                X_train.append(img)
                y_train.append(j)
                driver_id.append(driver_data[flbase])
                k+=1

        print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
        unique_drivers = sorted(list(set(driver_id)))
        print('Unique drivers: {}'.format(len(unique_drivers)))
        print(unique_drivers)
        return X_train, y_train, driver_id, unique_drivers


    #def load_test(self.img_rows, self.img_cols, color_type=1):
    def load_test(self, color_type=3):
        print('Read test images')
        start_time = time.time()
        path = os.path.join('state-farm-distracted-driver-detection','imgs', 'test', '*.jpg')
        files = glob.glob(path)
        X_test = []
        X_test_id = []
        total = 0
        thr = math.floor(len(files) / 10)
        k = 0
        for fl in tqdm(files):
            if k>100:
                break
            flbase = os.path.basename(fl)
            img = self.get_im_cv2_mod(fl,color_type)
            X_test.append(img)
            X_test_id.append(flbase)
            total += 1
            if total % thr == 0:
                print('Read {} images from {}'.format(total, len(files)))
            k+=1

        print('Read test data time: {} seconds'.format(round(time.time() - start_time, 2)))
        return X_test, X_test_id


    def cache_data(self,data, path):
        if os.path.isdir(os.path.dirname(path)):
            file = open(path, 'wb')
            pickle.dump(data, file)
            file.close()
        else:
            print('Directory doesnt exists')


    def restore_data(self,path):
        data = dict()
        if os.path.isfile(path):
            file = open(path, 'rb')
            data = pickle.load(file)
        return data


    def split_validation_set(self,train, target, test_size):
        random_state = 51
        X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test


    def read_and_normalize_train_data(self, color_type=3):
        cache_path = os.path.join('cache',
                                'train_r_' + str(self.img_rows) + '_c_' + str(self.img_cols) + '_t_' + str(color_type) + '.dat')
        
        if not os.path.isfile(cache_path) or use_cache == 0:
            train_data, train_target, driver_id, unique_drivers = self.load_train( color_type)
            self.cache_data((train_data, train_target, driver_id, unique_drivers), cache_path)
        else:
            print('Restore train from cache!')
            (train_data, train_target, driver_id, unique_drivers) = self.restore_data(cache_path)

        train_data = np.array(train_data, dtype=np.uint8)
        train_target = np.array(train_target, dtype=np.uint8)

        #train_data = np.array(train_data, dtype=np.float32)
        #train_target = np.array(train_target, dtype=np.float32)
        #train_data = train_data.reshape(train_data.shape[0], color_type, self.img_rows, self.img_cols)
        #train_data = train_data.reshape(train_data.shape[0], color_type)
        
        train_target = np_utils.to_categorical(train_target, 10)
        train_data = train_data.astype('float32')

        if self.normalization == '0_1':
            train_data /= 255
        elif self.normalization == 'standardzation':
            self.STD = train_data.std()
            self.MEAN = train_data.mean()
            train_data = (train_data-train_data.mean())/train_data.std()

        return train_data, train_target, driver_id, unique_drivers


    #def read_and_normalize_test_data(self.img_rows, self.img_cols, color_type=1):
    def read_and_normalize_test_data(self, color_type=3):

        cache_path = os.path.join('cache',
                                'test_r_' + str(self.img_rows) + '_c_' + str(self.img_cols) + '_t_' + str(color_type) + '.dat')
        if not os.path.isfile(cache_path) or use_cache == 0:
            test_data, test_id = self.load_test(color_type)
            self.cache_data((test_data, test_id), cache_path)
        else:
            print('Restore test from cache!')
            (test_data, test_id) = self.restore_data(cache_path)

        test_data = np.array(test_data, dtype=np.uint8)
        test_data = test_data.reshape(test_data.shape[0], self.img_rows, self.img_cols, color_type)
        test_data = test_data.astype('float32')
        
        if self.normalization == '0_1':
            test_data /= 255
        elif self.normalization == 'standardzation':
            self.STD = test_data.std()
            self.MEAN = test_data.mean()
            train_data = (test_data-test_data.mean())/test_data.std()

        return test_data, test_id


    def copy_selected_drivers(self,
                        train_data, 
                        train_target, 
                        driver_id, 
                        driver_list):

        data = []
        target = []
        index = []
        for i in range(len(driver_id)):
            if driver_id[i] in driver_list:
                data.append(train_data[i])
                target.append(train_target[i])
                index.append(i)
        data = np.array(data, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        index = np.array(index, dtype=np.uint32)

        return data, target, index

    def generate(self,p):
        print("self.img_cols", p[0][0])
        print("self.img_rows", p[0][1])

        if self.img_cols == p[0][0] and self.img_rows == p[0][1] and p[1] == self.normalization:
            
            return  self.X_train,          \
                    self.Y_train,          \
                    self.unique_list_train,\
                    self.X_valid,          \
                    self.Y_valid,          \
                    self.unique_list_valid,\
                    self.test_data,        \
                    self.test_id
        
        self.img_cols      = p[0][0]
        self.img_rows      = p[0][1]
        self.normalization = p[1]

        yfull_train = dict()
        yfull_test = []
        self.unique_list_train = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024',
                            'p026', 'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049',
                            'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p072',
                            'p075']
        self.unique_list_valid = ['p081']

        train_data, train_target, driver_id, unique_drivers = self.read_and_normalize_train_data()
        self.test_data, self.test_id = self.read_and_normalize_test_data()

        self.X_train, self.Y_train, self.train_index = self.copy_selected_drivers(train_data, train_target, driver_id, self.unique_list_train)        
        self.X_valid, self.Y_valid, self.test_index  = self.copy_selected_drivers(train_data, train_target, driver_id, self.unique_list_valid)

        
        print('Split train: ', len(self.X_train), len(self.Y_train))
        print('Split valid: ', len(self.X_valid), len(self.Y_valid))
        print('Train drivers: ', self.unique_list_train)
        print('Test drivers: ', self.unique_list_valid)

        return  self.X_train,          \
                self.Y_train,          \
                self.unique_list_train,\
                self.X_valid,          \
                self.Y_valid,          \
                self.unique_list_valid,\
                self.test_data,        \
                self.test_id


