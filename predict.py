from sklearn import metrics
from keras.models import load_model
import cv2
import tensorflow as tf
import os
import numpy as np
import os
import keras.backend as K


K.set_learning_phase(1)
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.65  # CUDA_ERROR_OUT_OF_MEMORY
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
modelpath='C:/Users/Administrator/Desktop/实验数据/训练好的模型/AtrousDense-U加厚-DeconvNet.h5'
prepath='D:/RemotSensing/testenhance/AtrousDenseheavyUDeconvNet/'
testsrc='D:/RemotSensing/testenhance/src/'
def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img, dtype="float") / 255.0
    return img
if __name__=='__main__':

 model=load_model(modelpath);
 files = os.listdir(testsrc)  # 得到文件夹下的所有文件名称
 for k in range(0,len(files),2):
     pre = np.zeros([2, 256, 256])
     testx1=load_img(os.path.join(testsrc, str(k)+'.tif'),False)
     testx2 = load_img(os.path.join(testsrc, str(k+1) + '.tif'), False)
     test=[]
     test.append(testx1)
     test.append(testx2)
     test=np.asarray(test)
     test_pred_prob_y = model.predict(test)
     for s in range(2):
       for i in range(256):
         for j in range(256):
             m = 0
             # print('c=',c,'w=',w)
             mv = test_pred_prob_y[s][i][j][0]
             for d in range(1, 6):
                 if test_pred_prob_y[s][i][j][d] > mv:
                     m = d;
                     mv = test_pred_prob_y[s][i][j][d]
             if m == 0:
                 pre[s][i][j] = 29
             if m == 1:
                 pre[s][i][j] = 76
             if m == 2:
                 pre[s][i][j] = 150
             if m == 3:
                 pre[s][i][j] = 179
             if m == 4:
                 pre[s][i][j] = 226
             if m == 5:
                 pre[s][i][j] = 255
     cv2.imwrite(prepath + str(k) + '.tif', pre[0])
     cv2.imwrite(prepath + str(k+1) + '.tif', pre[1])

 '''
 for l in range(0,10):
  w=1000*l
  test = []
  pre = np.zeros([1000, 256, 256])
  for c in range(w,w+1000):  # 遍历文件夹
    if c==len(files):
        break

    testx=load_img(os.path.join(testsrc, str(c)+'.tif'),False)
    test.append(testx)
    test1=np.asarray(test)

  test_pred_prob_y = model.predict(test1)
  for i in range(256):
        for j in range(256):
            m = 0
            #print('c=',c,'w=',w)
            mv = test_pred_prob_y[c-w][i][j][0]
            for k in range(1, 6):
                if test_pred_prob_y[c-w][i][j][k]> mv:
                    m = k;
                    mv = test_pred_prob_y[c-w][i][j][k]
            if m == 0:
                pre[c-w][i][j] = 29
            if m == 1:
                pre[c-w][i][j] = 76
            if m == 2:
                pre[c-w][i][j] = 150
            if m == 3:
                pre[c-w][i][j] = 179
            if m == 4:
                pre[c-w][i][j] = 226
            if m == 5:
                pre[c-w][i][j] = 255
  for c in range(w,w+1000):  # 遍历文件夹
    if c==len(files):
        break
    cv2.imwrite(prepath + str(c)+'.tif', pre[c-w])
    #cv2.imwrite(prepath   + '2939.tif', pre[1])
    #print(metrics.roc_auc_score(np.argmax(test_y0, axis=-1), np.argmax(pre[0], axis=-1)))
    #print(metrics.roc_auc_score(np.argmax(test_y1, axis=-1), np.argmax(pre[1], axis=-1)))
  '''