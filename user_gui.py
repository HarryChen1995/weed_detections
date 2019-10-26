import sys
import time
import json
import os, ssl
with open("weeds.json","r") as jfile:
    weed=json.loads(jfile.read())

import os
import numpy as np
from googlesearch import search
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QThread, pyqtSignal,pyqtSlot,Qt,QSize,QUrl
from PyQt5.QtWidgets import QApplication, QSizePolicy,QDialog, QWidget,QProgressBar,QPushButton,QToolTip,QFileDialog,QInputDialog,QLabel
from PyQt5.QtGui import QIcon, QFont,QPixmap,QImage,QMovie
from Predict_Image import predict
import cv2
#thread to call wrapper
class Thread(QThread):
    signal = pyqtSignal(tuple)
    def __init__(self,image_path):
        super().__init__()
        self.image_path=image_path

    def run(self):
        result=predict(self.image_path)
        self.signal.emit(result)

#web browser window
class web_windows(QWebEngineView):
    def __init__(self):
        super(web_windows,self).__init__()
        self.InitGUI()



    def InitGUI(self):
        self.setGeometry(200,500,1200,700)
# Main window 
class Windows(QWidget):
    def __init__(self):
        super(Windows, self).__init__()
        self.InitGUI()
    # initialzed all widgets 
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
        self.button3.resize(175,50)
        self.button3.move(700,660)
        self.button3.setIcon(QIcon("recognize.ico"))
        self.button3.setIconSize(QSize(24,24))
        self.button3.setShortcut('shift+R')
        self.button3.setStyleSheet("QPushButton { background-color: lightgray }"
                      "QPushButton:pressed { background-color: gray }" )
        self.button3.clicked.connect(self.start_thread)

        
        self.search_button=QPushButton('Google Search',self)
        self.search_button.setToolTip("Shortcut <b>shift+W</b>")
        self.search_button.resize(175,50)
        self.search_button.move(500,660)
        self.search_button.setIcon(QIcon("search.ico"))
        self.search_button.setIconSize(QSize(24,24))
        self.search_button.setShortcut('shift+W')
        self.search_button.setStyleSheet("QPushButton { background-color: lightgray }"
                      "QPushButton:pressed { background-color: gray }" )
        self.search_button.hide()
        self.web_page = web_windows()
        self.search_button.clicked.connect(self.search)
        self.setGeometry(500,500,1000,720)
        self.setWindowTitle('Weeds Detection Software')
        self.setWindowIcon(QIcon('icon.jpg'))
        self.setStyleSheet(open('stylesheet.css').read())
        self.show()
    # call back function for upload buttons
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
            self.search_button.hide()
        
        
        else:
            self.label4.setText("<b>Failed to Uploade Image!!!<\b>")
    #start thread and call wrapper function 
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
    # display output data from wrapper function
    def Finish(self,val):
        string =""
        self.web_title=""
        if len(val[1]) > 0:
            for weed_id, score in val[1].items():
                string += "<b>"+weed[weed_id]["name"]+"</b>: "+str(round(score*100,2))+"%<br>"
                self.web_title +=weed[weed_id]["name"]+" ,"
                string += "<b>the chemical killer</b>:<br>"+weed[weed_id]["action"]+"<br><br>"
            self.label6.setText(string)
            self.search_button.show()
        self.finish_image=val[0]
        height, width, _ = self.finish_image.shape
        bgra = np.zeros([height, width, 4], dtype=np.uint8)
        bgra[:, :, 0:3] = self.finish_image
        qimg = QImage(bgra.data, width, height,QImage.Format_RGB32)
        pixmap3=QPixmap.fromImage(qimg)
        self.label5.setPixmap(pixmap3)
        self.label5.setScaledContents(True)
        self.progress.setRange(0,1)
        self.progress.setValue(1)
        self.button3.setText("Done")
        self.button3.setIcon(QIcon("complete.ico"))
        self.button3.setShortcut('shift+R')
        self.label4.setText("")
    # save recognized image
    def save(self):
        if self.button3.text() =="Done":
            image,ok= QFileDialog.getSaveFileName(self, 'save image', '/home',"Images (*.png *.jpg *jpeg)")
            if ok:
                cv2.imwrite(image,self.finish_image)
    
    # run google search Engine
    def search(self):
        if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)): 
            ssl._create_default_https_context = ssl._create_unverified_context
        url=[i for i in search("How to kill and prevent "+self.web_title[:-2]+" weeds", tld='com', lang='en', num=1, stop=1, pause=0.1)]
        self.web_page.setWindowTitle("Google Search "+self.web_title[:-2])
        self.web_page.load(QUrl(url[0]))
        self.web_page.show()


if __name__  == "__main__":
    # launch GUI window
    app= QApplication([])
    ex = Windows()
    ex.setAttribute(Qt.WA_StyledBackground) 
    app.exec_()