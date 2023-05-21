import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from matplotlib import pyplot as plt
import imutils
import numpy as np
import easyocr
import cv2
from lp_detection import lp_detection

class VideoThread(QThread):
    ImageUpdate = pyqtSignal(QImage, list, list, list)

    def __init__(self, ocr_type, frame_skip, cam = True, path = ""):
        super().__init__()

        self.isCamera = cam
        self.vidPath = path
        self.OCR_type = ocr_type
        self.paused = False
        self.frame_skip_nr = frame_skip
        self.count = 0

    def run(self):
        self.ThreadActive = True
        if self.isCamera == True:
            capture = cv2.VideoCapture(0)
        else:
            capture = cv2.VideoCapture(self.vidPath)

        self.frame_rate = capture.get(cv2.CAP_PROP_FPS)
        
        while self.ThreadActive:
            
            while self.paused:
                pass

            success, frame = capture.read()
            if success:       
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ConvertToQtFormat = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                pic = ConvertToQtFormat.scaled(1280, 720, Qt.KeepAspectRatio)

                if self.count == self.frame_skip_nr:
                    (cam_img_res, cam_number_res, conf_res, _) = lp_detection(image, self.OCR_type, video=True)
                    self.count = 0
                else:
                    cam_img_res = ["__fail__"]
                    cam_number_res = []
                    conf_res = [0]

                    self.count += 1

                self.ImageUpdate.emit(pic, cam_img_res, cam_number_res, conf_res)

                cv2.waitKey(int(1000/self.frame_rate))

        capture.release()

    def stop(self):
        self.ThreadActive = False
        self.quit()

    def setPause(self):
        self.paused = not self.paused

        
class VideoFootage(QWidget):
    Pause = pyqtSignal(bool)

    def __init__(self, ocr_type, frame_skip, camera=True, vidPath = ""):
        super(VideoFootage, self).__init__()

        self.is_pause = False

        self.currentFrameNr = 0
        self.frameCache = []

        self.setObjectName("VideoFootageWidget")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("QWidget#VideoFootageWidget{ \n"
        "background-color:qlineargradient(spread:pad, x1:0.945, y1:0.0681818, x2:0, y2:1, stop:0 rgba(68, 206, 206, 255), stop:1 rgba(175, 219, 220, 255));}")

        self.mainLayout = QGridLayout()
        
        self.title = QLabel()
        if camera==True:
            self.title.setText("Camera")
        else:
            self.title.setText("Video")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setFont(QFont("Segoe UI", 12))
        self.mainLayout.addWidget(self.title, 0, 15, 1, 11)
        
        self.FeedLabel = QLabel()
        self.FeedLabel.setMaximumWidth(1100)
        self.FeedLabel.setMaximumHeight(700)
        self.mainLayout.addWidget(self.FeedLabel, 1, 16, 8, 11)
        
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setRowCount(0)
        self.table.setHorizontalHeaderLabels(['Plate Number', 'Confidence', 'Option'])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        
        self.mainLayout.addWidget(self.table, 1, 1, 1, 15)

        self.cancelBTN = QPushButton("Cancel")
        self.cancelBTN.clicked.connect(self.CancelFeed)
        self.cancelBTN.setFont(QFont("Segoe UI", 14))
        self.mainLayout.addWidget(self.cancelBTN, 10, 8, 1, 10)
        
        self.cancelBTN.setFixedWidth(600)
        self.cancelBTN.setFixedHeight(40)

        if camera==False:
            self.pauseBTN = QPushButton("Pause")
            self.pauseBTN.clicked.connect(self.pauseVideo)
            self.pauseBTN.setFont(QFont("Segoe UI", 14))
            self.mainLayout.addWidget(self.pauseBTN, 10, 20, 1, 10)
        
            self.pauseBTN.setFixedWidth(600)
            self.pauseBTN.setFixedHeight(40)

        self.setLayout(self.mainLayout)
    
        self.VideoThread = VideoThread(ocr_type, frame_skip, camera, vidPath)
        self.VideoThread.start()
        self.VideoThread.ImageUpdate.connect(self.updateScreen)


    @pyqtSlot(QImage, list, list, list)
    def updateScreen(self, image, cam_img_res, cam_number_res, conf_res):
        self.FeedLabel.setPixmap(QPixmap.fromImage(image))

        valid = 1

        if len(cam_number_res)>0 and cam_number_res!="__fail__":                  #=1; poate primi doar un numar
            ConvertToQtFormat = QImage(cam_img_res[0].data, cam_img_res[0].shape[1], cam_img_res[0].shape[0], QImage.Format_RGB888)
            pic = ConvertToQtFormat.scaled(1280, 720, Qt.KeepAspectRatio)
            self.FeedLabel.setPixmap(QPixmap.fromImage(pic))

            self.frameCache.append(cam_img_res[0])
            self.currentFrameNr = self.currentFrameNr + 1

            rowPosition = self.table.rowCount()

            for i in range(rowPosition):
                if cam_number_res[0] == self.table.item(i, 0).text():
                    valid = 0
                    break

            if valid == 1:
                self.table.insertRow(rowPosition)
                self.table.setItem(rowPosition, 0, QTableWidgetItem(cam_number_res[0]))
                if round(conf_res[0], 2) == 0:
                    self.table.setItem(rowPosition, 1, QTableWidgetItem("-"))
                else:
                    self.table.setItem(rowPosition, 1, QTableWidgetItem(str(round(conf_res[0], 2))+"%"))

                self.table.setItem(rowPosition, 2, QTableWidgetItem())

                self.saveButton =  QPushButton('Save frame', self)            
                self.saveButton.clicked.connect(lambda ch, i=self.currentFrameNr-1: self.saveFrame(i))

                self.table.setCellWidget(rowPosition, 2, self.saveButton)

    def pauseVideo(self):
        self.is_pause = not self.is_pause
        if self.is_pause:
            self.pauseBTN.setText("Play")
        else:
            self.pauseBTN.setText("Pause")
        self.VideoThread.setPause()

    def saveFrame(self, i):
        if(self.title.text()=="Video" and not self.is_pause):
            self.pauseVideo()
        cv2.imwrite("vid/frame"+str(i)+".jpg", cv2.cvtColor(self.frameCache[i], cv2.COLOR_BGR2RGB))
        if  self.frameCache[i].shape[0] < 500:
            temp = cv2.resize(self.frameCache[i], (2*self.frameCache[i].shape[1], 2*self.frameCache[i].shape[0]), cv2.INTER_LINEAR)
        else:
            temp = self.frameCache[i]
        cv2.imshow("CAR", cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))
        #cv2.imshow("CAR", cv2.cvtColor(self.frameCache[i], cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
  
    def CancelFeed(self):
        self.VideoThread.stop()
        stack.removeWidget(stack.currentWidget())

        
class ImageInput(QWidget):
    def __init__(self, path, ocr_type):
        super(ImageInput, self).__init__()
        
        self.setOCRType(ocr_type)
        self.setPath(path)
        self.image_step = 1

        self.setObjectName("ImageInputWidget")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("QWidget#ImageInputWidget{ \n"
        "background-color:qlineargradient(spread:pad, x1:0.945, y1:0.0681818, x2:0, y2:1, stop:0 rgba(68, 206, 206, 255), stop:1 rgba(175, 219, 220, 255));}")

        self.mainLayout = QGridLayout()
        self.mainLayout.setColumnStretch(0, 1)
        self.mainLayout.setColumnStretch(1, 1)
        self.mainLayout.setColumnStretch(2, 1)
        
        self.getResults()

        for i in range(len(self.nr_res)):
            rowPosition = self.table.rowCount()
            self.table.insertRow(rowPosition)
            self.table.setItem(rowPosition , 0, QTableWidgetItem(self.nr_res[i]))
            if round(self.conf_res[i], 2) == 0:
                self.table.setItem(rowPosition, 1, QTableWidgetItem("-"))
            else:
                self.table.setItem(rowPosition, 1, QTableWidgetItem(str(round(self.conf_res[i], 2))+"%"))

            self.table.setItem(rowPosition, 2, QTableWidgetItem())

            self.carButton = QPushButton('View car', self)            
            self.carButton.clicked.connect(lambda ch, i=i: self.showCarImage(i))

            self.table.setCellWidget(rowPosition, 2, self.carButton)
            

        if len(self.nr_res)>0:
            self.mainLayout.addWidget(self.table, 1, 1, 1, 10)


        if len(self.nr_res)>0:
            self.combo_details = QComboBox()
            self.combo_details.setStyleSheet("selection-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(255, 178, 102, 255), stop:0.55 rgba(235, 148, 61, 255), stop:0.98 rgba(0, 0, 0, 255), stop:1 rgba(0, 0, 0, 0));\n"
    "background-color: rgb(185, 222, 227);")
            self.combo_details.addItem("1. Grayscale")
            self.combo_details.addItem("2. Blackhat")
            self.combo_details.addItem("3. Sobel Edge")
            self.combo_details.addItem("4. Sobel Edge + Closing")
            self.combo_details.addItem("5. Sobel Edge + Threshold")
            self.combo_details.addItem("6. Sobel Edge + Threshold + Erosion/Dilation")
            self.combo_details.addItem("7. Light + Threshold")
            self.combo_details.addItem("8. Applied Mask")
            self.combo_details.addItem("9. Final")
            self.combo_details.addItem("10.Region of interest")
            self.combo_details.activated[str].connect(self.set_image_step)
            self.mainLayout.addWidget(self.combo_details, 6, 2, 1, 1)

            self.row_nr = QSpinBox()
            self.row_nr.setMaximum(self.table.rowCount())
            self.row_nr.setMinimum(1)
            self.mainLayout.addWidget(self.row_nr, 5, 2, 1, 1)

            self.showBTN = QPushButton("Show step")
            self.showBTN.clicked.connect(self.showStepImage)
            self.showBTN.setFont(QFont("Segoe UI", 14))
            self.mainLayout.addWidget(self.showBTN, 11, 2, 1, 1)

        self.backBTN = QPushButton("Back to Menu")
        self.backBTN.clicked.connect(self.returnToMenu)
        self.backBTN.setFont(QFont("Segoe UI", 14))
        self.mainLayout.addWidget(self.backBTN, 11, 1, 1, 1)
        
        self.backBTN.setMaximumWidth(600)
        self.backBTN.setMinimumHeight(40)

        self.setLayout(self.mainLayout)

    def set_image_step(self, text):
        self.image_step = text.split('.')[0]

    def showStepImage(self):
        h, w = self.steps_img[self.row_nr.value()-1][int(self.image_step)-1].shape
        self.temp = self.steps_img[self.row_nr.value()-1][int(self.image_step)-1]

        if h < 500:
            scale = 2
        else:
            scale = 1.1
        
        new_width = int(scale * w)
        new_height = int(scale * h)
        new_res = (new_width, new_height)
    
        self.temp = cv2.resize(self.temp, new_res, interpolation=cv2.INTER_LINEAR)

        cv2.imshow("CAR", self.temp)
        cv2.waitKey(0)

    def showCarImage(self, ind):
        h, w, _ = self.img_res[ind].shape
        self.temp = self.img_res[ind]

        if h < 450:
            scale = 2
        elif h < 350:
            scale = 3
        elif h > 950:
            scale = 0.6
        else:
            scale = 1.1
        
        new_width = int(scale * w)
        new_height = int(scale * h)
        new_res = (new_width, new_height)
    
        self.temp = cv2.resize(self.temp, new_res, interpolation=cv2.INTER_LINEAR)

        cv2.imshow("CAR", self.temp)
        cv2.waitKey(0)

    def getResults(self):
        self.car_image = cv2.imread(self.imagePath)
        h, w, ch = self.car_image.shape
        bytesPerLine = ch * w

        self.original_img_label = QLabel()
        self.qt_image = cv2.cvtColor(self.car_image, cv2.COLOR_BGR2RGB)
        self.qt_image = QImage(self.qt_image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        self.pic = self.qt_image.scaled(600, 400, Qt.KeepAspectRatio)
        self.original_img_label.setPixmap(QPixmap.fromImage(self.pic))
        self.mainLayout.addWidget(self.original_img_label, 1, 0, 1, 1)

        [self.img_res, self.nr_res, self.conf_res, self.steps_img] = lp_detection(self.car_image, self.OCR_type, video=False)

        if len(self.nr_res)==0:
            self.errorMessage = QLabel()
            self.errorMessage.setText("No license plate found!")
            self.errorMessage.setAlignment(Qt.AlignCenter)
            self.errorMessage.setFont(QFont("Segoe UI", 12))

            self.mainLayout.addWidget(self.errorMessage, 1, 1, 1, 1)
        else:
            self.table = QTableWidget()
            self.table.setColumnCount(3)
            self.table.setRowCount(0)
            self.table.setHorizontalHeaderLabels(['Plate Number', 'Confidence', 'Option'])
            self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        
    def setPath(self, path):
        self.imagePath = path

    def setOCRType(self, ocr_type):
        self.OCR_type = ocr_type

    def returnToMenu(self):
        stack.removeWidget(stack.currentWidget())

          
class HomePage(QMainWindow):
    def __init__(self):
        super(HomePage, self).__init__()

        self.ocr_type = "Tesseract"

        #Set Dimensions
        self.width = QDesktopWidget().screenGeometry(-1).width()
        self.height = QDesktopWidget().screenGeometry(-1).height()
        self.setGeometry(100, 100, self.width//2, self.height//2)
        self.setObjectName("MainWindow")

        self.setStyleSheet("QMainWindow#MainWindow{ \n"
        "background-color:qlineargradient(spread:pad, x1:0.945, y1:0.0681818, x2:0, y2:1, stop:0 rgba(68, 206, 206, 255), stop:1 rgba(175, 219, 220, 255));}")
        
        self.central_widget = QWidget()               
        self.setCentralWidget(self.central_widget)
        self.label = QLabel(self.central_widget)
        self.label.setGeometry(520, 10, 600, 70)
        self.label.setStyleSheet("font: 500 20pt \"Segoe UI\"; color: rgb(252, 255, 255);\n border-color: rgb(85, 0, 127);")
        self.label.setText("Automatic Number Plate Recognition")
        self.label.setAlignment(Qt.AlignCenter)

        self.imageInputButton = QPushButton("Insert an image", self.central_widget)
        self.imageInputButton.setStyleSheet("border-radius: 7px; \n font: 11pt \"Segoe UI\";  color: rgb(252, 255, 255);\n background-color: rgb(170, 96, 255); QPushButton::pressed{ background-color: rgb(0, 0, 255);}")
        #self.pushButton.setGeometry(QtCore.QRect(320, 130, 201, 61))
        self.imageInputButton.setGeometry(675, 200, 300, 100)
        self.imageInputButton.clicked.connect(self.getImage)
        
        self.videoButton = QPushButton("Select video", self.central_widget)
        #self.videoButton.setGeometry(320, 210, 201, 61)
        self.videoButton.setGeometry(675, 350, 300, 100)
        self.videoButton.setStyleSheet("border-radius: 7px;\n font: 11pt \"Segoe UI\";  color: rgb(252, 255, 255);\n background-color: rgb(170, 96, 255);")
        self.videoButton.clicked.connect(self.getVideo)

        self.cameraButton = QPushButton("Use Camera Footage", self.central_widget)
        self.cameraButton.setGeometry(675, 500, 300, 100)
        self.cameraButton.setStyleSheet("border-radius: 7px;\n font: 11pt \"Segoe UI\";  color: rgb(252, 255, 255);\n background-color: rgb(170, 96, 255);")
        self.cameraButton.clicked.connect(self.changeToCamera)

        self.label_ocr = QLabel(self.central_widget)
        self.label_ocr.setGeometry(175, 750, 200, 50)
        self.label_ocr.setText("Select Mode(OCR)")
        self.label_ocr.setStyleSheet("font: 11pt \"Segoe UI\";")
        self.label_ocr.setAlignment(Qt.AlignCenter)

        self.comboBox_ocr = QComboBox(self.central_widget)
        self.comboBox_ocr.setGeometry(100, 800, 350, 40)
        self.comboBox_ocr.setStyleSheet("selection-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(255, 178, 102, 255), stop:0.55 rgba(235, 148, 61, 255), stop:0.98 rgba(0, 0, 0, 255), stop:1 rgba(0, 0, 0, 0));\n"
"background-color: rgb(185, 222, 227);")
        self.comboBox_ocr.addItem("Fast (Tesseract)")
        self.comboBox_ocr.addItem("Slow (easyOCR)")
        self.comboBox_ocr.activated[str].connect(self.set_ocr)

        self.label_frame = QLabel("Frame skip:", self.central_widget)
        self.label_frame.setGeometry(1300, 800, 90, 30)
        self.label_frame.setStyleSheet("font: 11pt \"Segoe UI\";")
        self.spinBox = QSpinBox(self.central_widget)
        self.spinBox.setGeometry(1400, 802, 60, 30)
        self.spinBox.setMaximum(300)
        

    def set_ocr(self, text):
        text = text.split("(")[-1]
        self.ocr_type = text[:len(text)-1]
        

    def changeToCamera(self):
        self.frame_skip = self.spinBox.value()
        camera = VideoFootage(self.ocr_type, self.frame_skip)
        stack.addWidget(camera)
        stack.setCurrentWidget(camera)
        
    def getImage(self):
        fname = QFileDialog.getOpenFileName(self, 'Select an image', '', "Images (*.jpg; *.bmp; *.jpeg; *.png);;All Files (*)")
        imagePath = fname[0]
        
        if imagePath !="":
            img = ImageInput(imagePath, self.ocr_type)
            stack.addWidget(img)
            stack.setCurrentWidget(img)
    
    def getVideo(self):
        vname = QFileDialog.getOpenFileName(self, 'Select a video', '', "Video (*.mp4);;All Files (*)")
        videoPath = vname[0]

        if videoPath !="":
            self.frame_skip = self.spinBox.value()
            vid = VideoFootage(self.ocr_type, self.frame_skip, False, videoPath)
            stack.addWidget(vid)
            stack.setCurrentWidget(vid)
             

if __name__ == "__main__":
    App = QApplication(sys.argv)
    stack = QStackedWidget()
    
    home = HomePage()
    stack.addWidget(home)

    stack.setWindowTitle("Automatic Number Plate Recognition")
    
    stack.setFixedWidth(1600)
    stack.setFixedHeight(900)
   
    stack.show()

    App.exec_()
    App.quit()