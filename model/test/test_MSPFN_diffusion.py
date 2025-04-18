"""
For each file there's 5 samples, we evaluate all of them and pick the one with the highest PSNR (cherry pick)

"""
import numpy as np
import scipy
import cv2
import os
import glob
import time
import tensorflow as tf
# import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from collections import defaultdict
sys.path.append('../')
sys.path.append('../../utils')
#from TEST_MSPFN import Model
from TEST_MSPFN_M17N1 import Model
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
is_training = tf.placeholder(tf.bool, [])
dataset_names = ["mag_and_phase", "mag_and_phase_all_channels", "mag_only", "mag_only_all_channels"]

TARGETED_CFG_SCALE = 0 

for dataset_name in dataset_names:

    # dataset_name = "median_filter_implementation2"
    print(f"Running {dataset_name}",)
    img_path = f'test_data/{dataset_name}'
    file = os.listdir(img_path)
    save_path = f'test_data/{dataset_name}_predict'

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    num = 1
    
    for idx, f in enumerate(file):

        image_id, _, cfg_scale, _ = f.split("_")
        print(idx, f)
        if int(cfg_scale) != TARGETED_CFG_SCALE:
            continue
        pic_path = os.path.join(img_path, f)
        file_name = f
        img = cv2.imread(pic_path)
        W, H = img.shape[:2]
        img = img / 127.5 - 1
        input_ = np.zeros((1, W, H, 3)) # 16
        input_[0] = img
        if num==1:
            W_1 = W
            H_1 = H
            x_rain = tf.placeholder(tf.float32, [1, W, H, 3])#None
            model = Model(x_rain, is_training, 1)#16
            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)
            saver = tf.train.Saver()
            saver.restore(sess, 'epoch44') #93
            vars_all=tf.trainable_variables()
            print ('Params:',np.sum([np.prod(v.get_shape().as_list()) for v in vars_all]))
            st_time=time.time()
            fake = sess.run(
                [model.imitation],
                feed_dict={x_rain: input_, is_training: False})
            ed_time=time.time()
            cost_time=ed_time-st_time
            print('spent {} s.'.format(cost_time))
        else:
            if W_1==W and H_1 == H:
                st_time=time.time()
                fake = sess.run(
                    [model.imitation],
                    feed_dict={x_rain: input_, is_training: False})
                ed_time=time.time()
                cost_time=ed_time-st_time
                print('spent {} s.'.format(cost_time))
            else:
                #sess.close()
                W_1 = W
                H_1 = H
                x_rain = tf.placeholder(tf.float32, [1, W, H, 3])#None
                model = Model(x_rain, is_training, 1)#16
                init = tf.global_variables_initializer()
                sess = tf.Session()
                sess.run(init)
                saver = tf.train.Saver()
                saver.restore(sess, 'epoch44') #93
                st_time=time.time()
                fake = sess.run(
                    [model.imitation],
                    feed_dict={x_rain: input_, is_training: False})
                ed_time=time.time()
                cost_time=ed_time-st_time
                print('spent {} s.'.format(cost_time))
        img = fake[0]
        im = np.uint8(np.clip((img[0]+1)*127.5,0,255.0))
        cv2.imwrite(os.path.join(save_path, file_name), im)
        num+=1
