import sys
import time
import json
with open("weeds.json","r") as jfile:
    weed=json.loads(jfile.read())

import os
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal,pyqtSlot,Qt,QSize
from PyQt5.QtWidgets import QApplication, QSizePolicy,QDialog, QWidget,QProgressBar,QPushButton,QToolTip,QFileDialog,QInputDialog,QLabel
from PyQt5.QtGui import QIcon, QFont,QPixmap,QImage,QMovie
from Predict_Image import predict


import cv2

class Thread(QThread):
    signal = pyqtSignal(tuple)
    def __init__(self,image_path):
        super().__init__()
        self.image_path=image_path

    def run(self):
        result=predict(self.image_path)
        self.signal.emit(result)




class Windows(QWidget):
    def __init__(self):
        super(Windows, self).__init__()
        self.InitGUI()

    def InitGUI(self):
        QToolTip.setFont(QFont("SansSerif",10))
        label1 = QLabel(self)
        pixmap1 = QPixmap('logo.jpg')
        label1.setPixmap(pixmap1)
        label1.setScaledContents(True)
        label1.move(0,0)
        label1.resize(900,100)
        label2 = QLabel(self)
        pixmap2 = QPixmap('logo2.jpg')
        label2.setPixmap(pixmap2)
        label2.setScaledContents(True)
        label2.move(900,0)
        label2.resize(100,100)
        self.label5 = QLabel(self)
        self.label5.move(0,180)
        self.label5.setToolTip("Display Recognized Image Here")
        self.label5.resize(650,400)



        self.label3 = QLabel(self)
        self.label3.setText("<b>Progress<\b>")
        self.label3.move(450,580)
        self.label3.setStyleSheet("QLabel {background-color: rgba(196, 199, 202, 0.856)}")
        self.label3.resize(70,30)


        self.label4 = QLabel(self)
        self.label4.move(200,120)
        self.label4.setStyleSheet("QLabel {background-color: rgba(196, 199, 202, 0.856)}")
        self.label4.resize(230,30)

        self.progress = QProgressBar(self)
        self.progress.setGeometry(50, 620, 900, 30)
        self.progress.setMaximum(100)

        self.label7=QLabel("<b>Results:<b>",self)
        self.label7.setGeometry(670,165,70,20)
        self.label7.setStyleSheet("QLabel {  background-color: rgba(196, 199, 202, 0.856)}")
        
        self.label6 = QLabel(self)
        self.label6.setGeometry(660, 180, 325, 400)
        self.label6.setToolTip("Recognition Scores")
        self.label6.setStyleSheet("QLabel {  background-color: rgba(196, 199, 202, 0.856);\
             border: 5px solid rgb(95, 94, 94);\
             border-radius:7px;\
             font-size: 12px;\
             border-style:outset;\
             margin:5px}")
    



        button1=QPushButton('Upload Image',self)
        button1.setToolTip("Shortcut <b>shift+O</b>")
        button1.resize(150,50)
        button1.move(500,110)
        button1.setShortcut('shift+O')
        button1.setIcon(QIcon("upload.ico"))
        button1.setIconSize(QSize(24,24))
        button1.setStyleSheet("QPushButton { background-color: lightgray }"
                      "QPushButton:pressed { background-color: gray }" )
        button1.clicked.connect(self.upload)
        
        button2=QPushButton('Save Image',self)
        button2.setToolTip("Shortcut <b>shift+S</b>")
        button2.resize(150,50)
        button2.move(700,110)
        button2.setShortcut('shift+S')
        button2.setIcon(QIcon("download.ico"))
        button2.setIconSize(QSize(24,24))
        button2.setStyleSheet("QPushButton { background-color: lightgray }"
                      "QPushButton:pressed { background-color: gray }" )
        button2.clicked.connect(self.save)

        self.button3=QPushButton('Recognize Image',self)
        self.button3.setToolTip("Shortcut <b>shift+R</b>")
        self.button3.resize(150,50)
        self.button3.move(700,660)
        self.button3.setIcon(QIcon("recognize.ico"))
        self.button3.setIconSize(QSize(24,24))
        self.button3.setShortcut('shift+R')
        self.button3.setStyleSheet("QPushButton { background-color: lightgray }"
                      "QPushButton:pressed { background-color: gray }" )
        self.button3.clicked.connect(self.start_thread)


        self.setGeometry(500,500,1000,720)
        self.setWindowTitle('Weeds Detection Software')
        self.setWindowIcon(QIcon('icon.jpg'))
        self.setStyleSheet(open('stylesheet.css').read())
        self.show()

    def upload(self):
        self.image,_= QFileDialog.getOpenFileName(self, 'Open file', '/home',"Images (*.png *.jpg *jpeg)")
        
        if self.image:
            self.label4.setText("<b>Image Uploaded Successfully !!!<\b>")
            self.label6.setText("")
            pixmap3 = QPixmap(self.image)
            self.label5.setPixmap(pixmap3)
            self.label5.setScaledContents(True)
            self.button3.setText("Recognize Image")
            self.button3.setIcon(QIcon("recognize.ico"))
            self.button3.setShortcut('shift+R')
        
        
        else:
            self.label4.setText("<b>Failed to Uploade Image!!!<\b>")
    def start_thread(self):

        if self.label4.text() != "<b>Image Uploaded Successfully !!!<\b>":

            self.label4.setText("<b>Please to Upload Image !! </b>")
        else:
            self.Task = Thread(self.image)
            self.Task.signal.connect(self.Finish)
            self.progress.setRange(0,0)
            self.button3.setIcon(QIcon())
            self.button3.setText("working..")
            self.Task.start()
        
    def Finish(self,val):
        self.progress.setRange(0,1)
        self.progress.setValue(1)
        self.button3.setIcon(QIcon("complete.ico"))
        self.button3.setText("Done")
        self.button3.setShortcut('shift+R')
        self.label4.setText("")
        self.finish_image=val[0]
        height, width, _ = self.finish_image.shape
        bgra = np.zeros([height, width, 4], dtype=np.uint8)
        bgra[:, :, 0:3] = self.finish_image
        qimg = QImage(bgra.data, width, height,QImage.Format_RGB32)
        pixmap3=QPixmap.fromImage(qimg)
        self.label5.setPixmap(pixmap3)
        self.label5.setScaledContents(True)
        string =""
        if len(val[1]) > 0:
            for weed_id, score in val[1].items():
                string += "<b>"+weed[weed_id]["name"]+"</b>: "+str(round(score*100,2))+"%<br>"
                string += "<b>the chemical killer</b>:<br>"+weed[weed_id]["action"]+"<br><br>"
            self.label6.setText(string)


    def save(self):
        if self.button3.text() =="Done":
            fname, ok = QInputDialog.getText(self, 'Save File', 'enter file name(.jpg, .png, .jpeg):')
            if ok:
                cv2.imwrite(fname,self.finish_image)



        




if __name__  == "__main__":

    app= QApplication([])
    ex = Windows()
    ex.setAttribute(Qt.WA_StyledBackground) 
    app.exec_()
