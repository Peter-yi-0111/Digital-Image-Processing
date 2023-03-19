import sys
import torch
import os
import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
from networks.dinknet import DinkNet34
from tkinter import *
from tkinter import filedialog
from ui import Ui_Form
from PyQt5.QtWidgets import QMainWindow, QApplication
from tqdm import tqdm


def load_data(root):
    global imgt1
    global imgt2
    global imgct
    global imgft
    global imgmn
    global inputimg #(20,2,512,512)
    global inputtarget #(20,3,512,512)

    PATH_T1=root+"/T1"
    PATH_T2=root+"/T2"
    PATH_CT=root+"/CT"
    PATH_FT=root+"/FT"
    PATH_MN=root+"/MN"

    #print(PATH_T1)

    imgt1 = []
    imgt2 = []
    imgct = []
    imgft = []
    imgmn = []

    imgt1 = read_data(PATH_T1)
    imgt2 = read_data(PATH_T2)
    imgct = read_data(PATH_CT)
    imgft = read_data(PATH_FT)
    imgmn = read_data(PATH_MN)
    #print(imgmn[19].shape)  #20張(512,512)

    temp = []
    inputimg = []
    inputtarget = []

    for i in range(len(imgt1)):
        temp.append(imgt1[i])
        temp.append(imgt2[i])
        inputimg.append(temp)
        temp = []
        temp.append(imgct[i])
        temp.append(imgft[i])
        temp.append(imgmn[i])
        inputtarget.append(temp)
        temp = []

    #print(np.array(inputtarget).shape) #(20,3,512,512)

    return 0

def read_data(path):
    array_of_image = []
    images = glob.glob(path+"\*.jpg")
    #print(image)
    for filename in images:
        img = cv2.imread(filename,0)
        #print(filename)
        array_of_image.append(img)
        #print(img)
    return array_of_image



class MainWindow(QMainWindow, Ui_Form):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()


    def onBindingUI(self):
        self.btn1.clicked.connect(self.on_btn1_click)
        self.btn2.clicked.connect(self.on_btn2_click)
        self.btn3.clicked.connect(self.on_btn3_click)

    def on_btn1_click(self):
        root = Tk()
        root.directory=  filedialog.askdirectory()
        #print (root.directory)  C:/Users/peter/Desktop/lab/hw/影像處理/hw2/carpalTunnel/0
        load_data(root.directory)
        print (root.directory)
        root.destroy()

        #test all data in folder
        global inputimg #dim = (20,2,512,512)
        global outputtargat #dim = (20,512,512,3)
        inputimg = np.array(inputimg)
        outputtargat = []
        for i in tqdm(range(np.size(inputimg,0))):
            inputmodule = inputimg[i]
            #print(np.array(inputmodule).shape) #(2,512,512)
            inputmodule = np.array(inputmodule)
            inputmodule = np.expand_dims(inputmodule,0)
            #print(np.array(inputmodule).shape) #(1,2,512,512)
            inputmodule = inputmodule / 255
            with torch.no_grad():
                device = torch.device("cuda")
                model = DinkNet34()
                model.cuda()
                model.load_state_dict(torch.load('model_dlinknet2.pth'))
                model.eval()
                inputmodule = torch.tensor(inputmodule)
                inputmodule = inputmodule.type(torch.FloatTensor).cuda()
                output = model(inputmodule)
                output = output.cpu()
                output = np.array(output)
                output =np.squeeze(output)
                #print(output.shape) #(3,512,512)
                output = np.transpose(output,(1,2,0)).round() #(3,512,512) 變成 (512,512,3)
                outputtargat.append(output)
        #print(np.array(outputtargat).shape) #(20,512,512,3)

    def on_btn2_click(self):
        global outputtargat #dim = (20,512,512,3)
        global inputimg #(20,2,512,512)
        global inputtarget #(20,3,512,512)

        img_num = self.cboxImgNum_1.currentText()
        outputtargat = np.array(outputtargat)
        #print(outputtargat[int(img_num)].shape)

        inputimg = np.array(inputimg)
        inputtarget = np.array(inputtarget)

        num_folder_input = inputimg[int(img_num)]
        num_folder_target = inputtarget[int(img_num)]

        num_folder_target = np.transpose(num_folder_target,(1,2,0)).round()

        cv2.imshow('T1',num_folder_input[0])
        cv2.imshow('T2',num_folder_input[1])
        cv2.imshow('module_output',outputtargat[int(img_num)])
        cv2.imshow('ground truth',num_folder_target)

        #edge detection
        outputtargat1 = np.transpose(outputtargat,(0,3,1,2))
        outputtargat1 = outputtargat1[int(img_num)]
        inputtarget1 = inputtarget[int(img_num)]

        #改成uint8
        outputtargat1 = np.uint8(outputtargat1)
        inputtargat1 = np.uint8(inputtarget1)

        #高斯模糊
        blur0 = cv2.GaussianBlur(outputtargat1[0],(9,9),0)
        blur1 = cv2.GaussianBlur(outputtargat1[1],(9,9),0)
        blur2 = cv2.GaussianBlur(outputtargat1[2],(9,9),0)

        blur3 = cv2.GaussianBlur(inputtargat1[0],(9,9),0)
        blur4 = cv2.GaussianBlur(inputtargat1[1],(9,9),0)
        blur5 = cv2.GaussianBlur(inputtargat1[2],(9,9),0)


        #canny
        canny0 = cv2.Canny(outputtargat1[0] * 255, 50, 150)
        canny1 = cv2.Canny(outputtargat1[1] * 255, 50, 150)
        canny2 = cv2.Canny(outputtargat1[2] * 255, 50, 150)

        canny3 = cv2.Canny(inputtargat1[0], 50, 150)
        canny4 = cv2.Canny(inputtargat1[1], 50, 150)
        canny5 = cv2.Canny(inputtargat1[2], 50, 150)

        #Finding Contours
        contours0, hierarchy0 = cv2.findContours(canny0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours1, hierarchy1 = cv2.findContours(canny1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours2, hierarchy2 = cv2.findContours(canny2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        contours3, hierarchy3 = cv2.findContours(canny3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours4, hierarchy4 = cv2.findContours(canny4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours5, hierarchy5 = cv2.findContours(canny5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        #draw
        inputimg1 = np.array(inputimg)
        inputimg1 = inputimg[int(img_num)][0]
        inputimg2 = inputimg[int(img_num)][0]

        inputimg1 = cv2.cvtColor(inputimg1,cv2.COLOR_GRAY2RGB)
        inputimg2 = cv2.cvtColor(inputimg2,cv2.COLOR_GRAY2RGB)

        cv2.drawContours(inputimg1, contours0, -1, (255, 0, 0), 2)
        cv2.drawContours(inputimg1, contours1, -1, (0, 255 , 0), 2)
        cv2.drawContours(inputimg1, contours2, -1, (0, 0, 255), 2)

        cv2.drawContours(inputimg2, contours3, -1, (255, 0, 0), 2)
        cv2.drawContours(inputimg2, contours4, -1, (0, 255 , 0), 2)
        cv2.drawContours(inputimg2, contours5, -1, (0, 0, 255), 2)

        cv2.imshow('final', inputimg1)
        cv2.imshow('final target', inputimg2)


    def on_btn3_click(self):
        global inputtarget #(20,3,512,512)
        global outputtargat #dim = (20,512,512,3)
        img_num = self.cboxImgNum_1.currentText()
        #sequence
        #inputtarget = np.array(inputtarget)
        inputtarget1 = np.transpose(inputtarget,(1,0,2,3))/255
        #print(inputtarget1.shape) #dim = (3,20,512,512)
        #outputtargat = np.array(outputtargat)
        outputtargat1 = np.transpose(outputtargat,(3,0,1,2))
        #print(outputtargat1.shape) #dim = (3,20,512,512)
        #print(inputtarget1[0].shape) #(20,512,512)
        #print(inputtarget1[0][1][256][220])
        ct_seq_acc = 0
        ft_seq_acc = 0
        mn_seq_acc = 0

        a = (2 * outputtargat1 * inputtarget1).sum((2,3)) #(3,18)
        b = (outputtargat1+inputtarget1).sum((2,3))
        #print(a[0])
        #rint(b[0])

        for i in range(np.size(inputtarget1,1)):

            ct_seq_acc += a[0][i]/b[0][i]
            ft_seq_acc += a[1][i]/b[1][i]
            mn_seq_acc += a[2][i]/b[2][i]

        ct_seq_acc = (ct_seq_acc / np.size(inputtarget1,1))
        ft_seq_acc = (ft_seq_acc / np.size(inputtarget1,1))
        mn_seq_acc = (mn_seq_acc / np.size(inputtarget1,1))
        #print(np.size(inputtarget1,1)) #圖片數量
        self.label_4.setText( "CT : {:.3f}".format(ct_seq_acc) )
        self.label_3.setText( "FT : {:.3f}".format(ft_seq_acc) )
        self.label_2.setText( "MN : {:.3f}".format(mn_seq_acc) )
        #print(ct_seq_acc)
        #print(ft_seq_acc)
        #print(mn_seq_acc)

        #單張圖

        ct_img_acc = a[0][int(img_num)]/b[0][int(img_num)]
        ft_img_acc = a[1][int(img_num)]/b[1][int(img_num)]
        mn_img_acc = a[2][int(img_num)]/b[2][int(img_num)]

        self.label_8.setText( "CT : {:.3f}".format(ct_img_acc) )
        self.label_7.setText( "FT : {:.3f}".format(ft_img_acc) )
        self.label_6.setText( "MN : {:.3f}".format(mn_img_acc) )

        '''for i in range(np.size(inputtarget,1)):
            a = (2 * outputtargat[i] * inputtarget[i]).sum((1,2))
            b = outputtargat[i].sum((2,3)) +inputtarget[i].sum((1,2))
            seq_acc += a/b

        seq_acc = seq_acc / np.size(inputtarget,0)
        print(seq_acc)
        img_num = self.cboxImgNum_1.currentText()'''

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
