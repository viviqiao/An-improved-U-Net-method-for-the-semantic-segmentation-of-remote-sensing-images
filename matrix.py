from sklearn.metrics import confusion_matrix
import cv2
import numpy as np
import os
import pandas as pd
label='D:/RemotSensing/testenhance/mask/'
def load_img(path, grayscale=True):
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img, dtype="float") / 255.0
    return img


def count(data):

    s1 = data.sum()
    # s2=data2.sum()
    FN = np.zeros([6])
    FP = np.zeros([6])
    TP = np.zeros([6])
    TN = np.zeros([6])
    TOP1=np.zeros([6])
    # 横为FN  竖为FP
    for i in range(6):
        s = 0;
        for j in range(6):
            if i != j:
                s += data[i][j]
            else:
                TP[i] = data[i][j]
        FN[i] = s
    for j in range(6):
        s = 0;
        for i in range(6):
            if i != j:
                s += data[i][j]
        FP[j] = s
    print(FN.sum(),FP.sum())
    for i in range(6):
        TN[i] = s1 - FN[i] - FP[i] - TP[i] - TN[i]

    TPR = np.zeros([6])
    TNR = np.zeros([6])
    PPV = np.zeros([6])
    NPV = np.zeros([6])
    ACC = np.zeros([6])
    IoU = np.zeros([6])
    F1 = np.zeros([6])
    # 评价指标：
    for i in range(6):
        TPR[i] = TP[i] / (TP[i] + FN[i])  # 召回率
        TNR[i] = TN[i] / (TN[i] + FP[i])
        PPV[i] = TP[i] / (TP[i] + FP[i])  # 精确率
        NPV[i] = TN[i] / (TN[i] + FN[i])
        ACC[i] = (TP[i] + TN[i]) / (TP[i] + FP[i] + FN[i] + TN[i])  # 准确率
        IoU[i] = TP[i] / (FN[i] + FP[i] + TP[i])  # IoU
        F1[i] = (2 * TPR[i] * PPV[i]) / (TPR[i] + PPV[i])

        print('class', i, ' IoU:', IoU[i])
        print('class', i, ' recall:', TPR[i])
        print('class', i, ' TNR:', TNR[i])
        print('class', i, ' precision:', PPV[i])
        print('class', i, ' NPV:', NPV[i])
        print('class', i, ' ACC:', ACC[i])
        print('class', i, ' F1:',F1[i])
        #print('class', i, ' F1:', F1[i])
    # 高级指标PA
    ac = 0
    for i in range(6):
        ac += TP[i]
    print('PA:', ac / s1)
    # 高级指标mPA
    c = 0
    for i in range(6):
        c += PPV[i]
    print('mPA:', c / 6)
    # 高级指标mIoU
    c = 0
    for i in range(6):
        c += IoU[i]
    print('mIoU:', c / 6)
    print('comprehensive:')
    P=(TP[0]+TP[1]+TP[2]+TP[3]+TP[4]+TP[5])/(TP[0]+TP[1]+TP[2]+TP[3]+TP[4]+TP[5]+FP[0]+FP[1]+FP[2]+FP[3]+FP[4]+FP[5])
    R=(TP[0]+TP[1]+TP[2]+TP[3]+TP[4]+TP[5])/(TP[0]+TP[1]+TP[2]+TP[3]+TP[4]+TP[5]+FN[0]+FN[1]+FN[2]+FN[3]+FN[4]+FN[5])
    f1=F1.sum()/6
    print('P:',P,'R:',R,'F1',f1)
if __name__=='__main__':


    data1 = np.load(r"C:\Users\Administrator\Desktop\实验数据\AtrousDenseheavyUDeconvNetmartrix.npy")
    data2 = np.load(r"C:\Users\Administrator\Desktop\实验数据\AtrousDenseUDeconvNetmatrix.npy")
    data3 = np.load(r"C:\Users\Administrator\Desktop\实验数据\AtrousDenseUDeconvUnetmatrix.npy")
    data4 = np.load(r"C:\Users\Administrator\Desktop\实验数据\ConvUConvNetmatrix.npy")
    data5 = np.load(r"C:\Users\Administrator\Desktop\实验数据\Unetmatrix.npy")


    count(data1)
    count(data2)
    count(data3)
    count(data4)
    count(data5)



    ''''
       print('class',i,' TPR:',TPR[i])
       print('class', i, ' TNR:',TNR[i])
       print('class', i, ' PPV:', PPV[i])
       print('class', i, ' NPV:', NPV[i])
       print('class', i, ' ACC:', ACC[i])



    files = os.listdir(label)  # 得到文件夹下的所有文件名称
    pre1=[]
    pre2=[]
    pre3=[]
    true=[]
    m1=np.zeros([6,6])
    m2=np.zeros([6,6])
    m3 = np.zeros([6, 6])
    for k in range(0, 20104):
        premask1 = load_img(r'D:/RemotSensing/testenhance/Unet/' + str(k) + '.tif')
        #premask2 = load_img('D:/RemotSensing/testenhance/AtrousDenseUDeconvUnet/' + str(k) + '.tif')
        #premask3 = load_img('D:/RemotSensing/testenhance/Unet/' + str(k) + '.tif')
        truemask = load_img(label + str(k) + '.tif')
        for i in range(256):
            for j in range(256):
                if truemask[i][j]==29:
                    if premask1[i][j]==29:
                        m1[0][0]+=1
                    if premask1[i][j] == 76:
                        m1[0][1] += 1
                    if premask1[i][j] == 150:
                        m1[0][2] += 1
                    if premask1[i][j] == 179:
                        m1[0][3] += 1
                    if premask1[i][j] == 226:
                        m1[0][4] += 1
                    if premask1[i][j] == 255:
                        m1[0][5] += 1
                if truemask[i][j] == 76:
                    if premask1[i][j]==29:
                        m1[1][0]+=1
                    if premask1[i][j] == 76:
                        m1[1][1] += 1
                    if premask1[i][j] == 150:
                        m1[1][2] += 1
                    if premask1[i][j] == 179:
                        m1[1][3] += 1
                    if premask1[i][j] == 226:
                        m1[1][4] += 1
                    if premask1[i][j] == 255:
                        m1[1][5] += 1
                if truemask[i][j] == 150:
                    if premask1[i][j]==29:
                        m1[2][0]+=1
                    if premask1[i][j] == 76:
                        m1[2][1] += 1
                    if premask1[i][j] == 150:
                        m1[2][2] += 1
                    if premask1[i][j] == 179:
                        m1[2][3] += 1
                    if premask1[i][j] == 226:
                        m1[2][4] += 1
                    if premask1[i][j] == 255:
                        m1[2][5] += 1
                if truemask[i][j] == 179:
                    if premask1[i][j]==29:
                        m1[3][0]+=1
                    if premask1[i][j] == 76:
                        m1[3][1] += 1
                    if premask1[i][j] == 150:
                        m1[3][2] += 1
                    if premask1[i][j] == 179:
                        m1[3][3] += 1
                    if premask1[i][j] == 226:
                        m1[3][4] += 1
                    if premask1[i][j] == 255:
                        m1[3][5] += 1
                if truemask[i][j] == 226:
                    if premask1[i][j]==29:
                        m1[4][0]+=1
                    if premask1[i][j] == 76:
                        m1[4][1] += 1
                    if premask1[i][j] == 150:
                        m1[4][2] += 1
                    if premask1[i][j] == 179:
                        m1[4][3] += 1
                    if premask1[i][j] == 226:
                        m1[4][4] += 1
                    if premask1[i][j] == 255:
                        m1[4][5] += 1
                if truemask[i][j] == 255:
                    if premask1[i][j]==29:
                        m1[5][0]+=1
                    if premask1[i][j] == 76:
                        m1[5][1] += 1
                    if premask1[i][j] == 150:
                        m1[5][2] += 1
                    if premask1[i][j] == 179:
                        m1[5][3] += 1
                    if premask1[i][j] == 226:
                        m1[5][4] += 1
                    if premask1[i][j] == 255:
                        m1[5][5] += 1
                                
    np.save('m1',m1)
        
        
        '''
        
        
        
        
''' 
        pre1.append(premask1)
        #pre2.append(premask2)
        #pre3.append(premask3)
        true.append(truemask)
    pre1 = np.asarray(pre1)
    #pre2 = np.asarray(pre2)
    #pre3 = np.asarray(pre3)
    true=np.asarray(true)
    # pre1 = pre1.apply(pd.to_numeric, errors='coerce')
    # pre2 = pre1.apply(pd.to_numeric, errors='coerce')
    #pre3 = pre1.apply(pd.to_numeric, errors='coerce')
    pre1 = np.resize(pre1, (20104*256*256))
    #pre2 = np.resize(pre2, (20104*256*256))
    #pre3 = np.resize(pre3, (20104*256*256))
    true=np.resize(true,(20104*256*256))

    m1=  confusion_matrix(true, pre1, labels=np.asarray([29,76,150,179,226,255],dytpe='unit8'), sample_weight=None)
    #m2 = confusion_matrix(true, pre2, labels=[29, 76, 150, 179, 226, 255], sample_weight=None)
    #m3 = confusion_matrix(true, pre3, labels=[29, 76, 150, 179, 226, 255], sample_weight=None)
    np.save('m1.txt',m1)
    #np.save('m2.txt', m1)
    #np.save('m3.txt', m1)
'''
