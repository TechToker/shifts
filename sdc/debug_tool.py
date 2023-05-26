import copy

from PyQt5 import QtCore, QtGui, QtWidgets

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import collections as mc
import cv2

from ysdc_dataset_api.dataset import MotionPredictionDataset
from ysdc_dataset_api.features import FeatureRenderer
from ysdc_dataset_api.utils import transform_2d_points

from scipy.ndimage.interpolation import rotate
import test as tst


SHIFT_STEP = 3
ROTATION_ANGLE = 4
MAP_ROI_SIZE = (128, 128)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(650, 490, 121, 41))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.onInferenceClick)

        self.canvas = QtWidgets.QLabel(self.centralwidget)
        self.canvas.setGeometry(QtCore.QRect(10, 10, 512, 512))
        self.canvas.setObjectName("label")

        self.preview = QtWidgets.QLabel(self.centralwidget)
        self.preview.setGeometry(QtCore.QRect(540, 10, 256, 256))
        self.preview.setObjectName("label")

        self.pushButton_down = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_down.setGeometry(QtCore.QRect(670, 460, 71, 25))
        self.pushButton_down.setObjectName("pushButton_down")
        self.pushButton_down.clicked.connect(self.onClickDown)

        self.pushButton_up = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_up.setGeometry(QtCore.QRect(670, 400, 71, 25))
        self.pushButton_up.setObjectName("pushButton_up")
        self.pushButton_up.clicked.connect(self.onClickUp)

        self.pushButton_right = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_right.setGeometry(QtCore.QRect(720, 430, 71, 25))
        self.pushButton_right.setObjectName("pushButton_right")
        self.pushButton_right.clicked.connect(self.onClickRight)

        self.pushButton_left = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_left.setGeometry(QtCore.QRect(630, 430, 71, 25))
        self.pushButton_left.setObjectName("pushButton_left")
        self.pushButton_left.clicked.connect(self.onClickLeft)

        self.pushButton_rotateClock = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_rotateClock.setGeometry(QtCore.QRect(720, 370, 71, 25))
        self.pushButton_rotateClock.setObjectName("pushButton_rotateClock")
        self.pushButton_rotateClock.clicked.connect(self.onRotateClock)

        self.pushButton_counterRotate = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_counterRotate.setGeometry(QtCore.QRect(630, 370, 71, 25))
        self.pushButton_counterRotate.setObjectName("pushButton_counterRotate")
        self.pushButton_counterRotate.clicked.connect(self.onRotateCounter)

        self.label_vel1 = QtWidgets.QLabel(self.centralwidget)
        self.label_vel1.setGeometry(QtCore.QRect(630, 270, 51, 17))
        self.label_vel1.setObjectName("label_vel1")
        self.label_vel2 = QtWidgets.QLabel(self.centralwidget)
        self.label_vel2.setGeometry(QtCore.QRect(630, 290, 67, 17))
        self.label_vel2.setObjectName("label_vel2")
        self.label_acc1 = QtWidgets.QLabel(self.centralwidget)
        self.label_acc1.setGeometry(QtCore.QRect(630, 310, 67, 17))
        self.label_acc1.setObjectName("label_acc1")
        self.label_acc2 = QtWidgets.QLabel(self.centralwidget)
        self.label_acc2.setGeometry(QtCore.QRect(630, 330, 67, 17))
        self.label_acc2.setObjectName("label_acc2")
        self.label_yaw = QtWidgets.QLabel(self.centralwidget)
        self.label_yaw.setGeometry(QtCore.QRect(630, 350, 67, 17))
        self.label_yaw.setObjectName("label_yaw")
        self.textEdit_vel1 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_vel1.setGeometry(QtCore.QRect(680, 270, 91, 21))
        self.textEdit_vel1.setObjectName("textEdit_vel1")
        self.textEdit_vel2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_vel2.setGeometry(QtCore.QRect(680, 290, 91, 21))
        self.textEdit_vel2.setObjectName("textEdit_vel2")
        self.textEdit_acc1 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_acc1.setGeometry(QtCore.QRect(680, 310, 91, 21))
        self.textEdit_acc1.setObjectName("textEdit_acc1")
        self.textEdit_acc2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_acc2.setGeometry(QtCore.QRect(680, 330, 91, 21))
        self.textEdit_acc2.setObjectName("textEdit_acc2")
        self.textEdit_yaw = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_yaw.setGeometry(QtCore.QRect(680, 350, 91, 21))
        self.textEdit_yaw.setObjectName("textEdit_yaw")


        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # sdc init

        with open('./sdc/renderer_config.yaml') as f:
            renderer_config = yaml.safe_load(f)

        renderer_config['feature_map_params']['rows'] = 256
        renderer_config['feature_map_params']['cols'] = 256

        renderer = FeatureRenderer(renderer_config)

        dataset_path = '../dataset/development_pb/'
        dataset = MotionPredictionDataset(
            dataset_path=dataset_path,
            feature_producers=[renderer],
            transform_ground_truth_to_agent_frame=True)

        # Scene search
        innopolis_scenes = []
        for d_item in dataset:
            if d_item['ground_truth_trajectory'][-1, 0] > 2.0 or d_item['ground_truth_trajectory'][-1, 1] > 2.0:
                # if d_item['scene_id'] == 'abbfe977229cd8f605de977495eee9f4':
                if d_item['scene_tags']['track'] == 'Innopolis':
                    innopolis_scenes.append(d_item)

                    print(len(innopolis_scenes))

                    if len(innopolis_scenes) == 105:
                        break

        self.data_item = innopolis_scenes[-1]

        self.current_feature_map = self.data_item['feature_maps']

        shape = self.current_feature_map[0].shape

        x_center = int(shape[1] / 2)
        y_center = int(shape[0] / 2)

        self.ego_features = np.zeros((6, 8, 8))

        for i in range(0, 6):  # go through Vehicle renderer channels
            self.ego_features[i] = copy.deepcopy(self.current_feature_map[i, y_center - 4:y_center + 4, x_center - 4:x_center + 4])
            self.current_feature_map[i, y_center - 4:y_center + 4, x_center - 4:x_center + 4] = 0

        # print('here')
        #print(np.amax(self.current_feature_map[0, y_center - 3:y_center + 3, x_center - 4:x_center + 4]))

        self.current_feature_map_w_taget = self.put_target_agent_on_map(self.current_feature_map)
        self.unrotated_map = copy.deepcopy(self.current_feature_map)
        self.map_roi = self.get_map_roi()

        self.visualization()
        self.rotation_angle = 0

        self.model = tst.Test()

        # for map in data_item['feature_maps']:
        #     plt.imshow(map, origin='lower', cmap='binary')
        #     plt.show()



    def get_map_roi(self):
        shape = self.current_feature_map_w_taget[0].shape

        x = int(shape[1] / 2 - MAP_ROI_SIZE[0] / 2)
        y = int(shape[0] / 2 - MAP_ROI_SIZE[0] / 2)

        return self.current_feature_map_w_taget[:, y:y+MAP_ROI_SIZE[0], x:x+MAP_ROI_SIZE[1]]

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Prediction debugger"))
        self.pushButton.setText(_translate("MainWindow", "Inference"))
        self.canvas.setText(_translate("MainWindow", "TextLabel"))
        self.pushButton_down.setText(_translate("MainWindow", "Down"))
        self.pushButton_up.setText(_translate("MainWindow", "Up"))
        self.pushButton_right.setText(_translate("MainWindow", "Right"))
        self.pushButton_left.setText(_translate("MainWindow", "Left"))
        self.pushButton_rotateClock.setText(_translate("MainWindow", "cl_rot"))
        self.pushButton_counterRotate.setText(_translate("MainWindow", "count_rot"))
        self.label_vel1.setText(_translate("MainWindow", "Vel1"))
        self.label_vel2.setText(_translate("MainWindow", "Vel2"))
        self.label_acc1.setText(_translate("MainWindow", "Acc1"))
        self.label_acc2.setText(_translate("MainWindow", "Acc2"))
        self.label_yaw.setText(_translate("MainWindow", "Yaw"))

    def onInferenceClick(self):
        print("inf")

        vel1 = float(self.textEdit_vel1.toPlainText())
        vel2 = float(self.textEdit_vel2.toPlainText())
        acc1 = float(self.textEdit_acc1.toPlainText())
        acc2 = float(self.textEdit_acc2.toPlainText())
        yaw = float(self.textEdit_yaw.toPlainText())

        conf = [vel1, vel2, acc1, acc2, yaw]

        for i in range(1, 6):
            self.ego_features[i][self.ego_features[i] != 0] = conf[i - 1]

        shape = self.map_roi[0].shape

        x_center = int(shape[1] / 2)
        y_center = int(shape[0] / 2)

        for i in range(0, 6):  # go through Vehicle renderer channels
            self.map_roi[i, y_center - 4:y_center + 4, x_center - 4:x_center + 4] = self.ego_features[i]

        self.data_item['prerendered_feature_map'] = self.map_roi
        img_res = self.model.inference(self.data_item)

        qpixmap_preview = self.np_array_to_pixmap(img_res)
        self.preview.setPixmap(qpixmap_preview)

    def put_target_agent_on_map(self, current_feature_map):
        #return current_feature_map

        shape = current_feature_map[0].shape

        x_center = int(shape[1] / 2)
        y_center = int(shape[0] / 2)

        target_agent_map = copy.deepcopy(current_feature_map)

        for i in range(0, 6):  # go through Vehicle renderer channels
            target_agent_map[i, y_center - 4:y_center + 4, x_center - 4:x_center + 4] = self.ego_features[i]

        # target_agent_map[0, y_center - 2:y_center + 2, x_center - 9:x_center - 5] = 1

        return target_agent_map

    def onClickRight(self):
        self.current_feature_map = np.roll(self.current_feature_map, SHIFT_STEP, axis=2)
        self.unrotated_map = copy.deepcopy(self.current_feature_map)
        self.onMapChange()

    def onClickLeft(self):
        self.current_feature_map = np.roll(self.current_feature_map, -SHIFT_STEP, axis=2)
        self.unrotated_map = copy.deepcopy(self.current_feature_map)
        self.onMapChange()

    def onClickUp(self):
        self.current_feature_map = np.roll(self.current_feature_map, SHIFT_STEP, axis=1)
        self.unrotated_map = copy.deepcopy(self.current_feature_map)
        self.onMapChange()

    def onClickDown(self):
        self.current_feature_map = np.roll(self.current_feature_map, -SHIFT_STEP, axis=1)
        self.unrotated_map = copy.deepcopy(self.current_feature_map)
        self.onMapChange()

    def onRotateCounter(self):
        self.rotation_angle += ROTATION_ANGLE
        self.rotate_map()

    def onRotateClock(self):
        self.rotation_angle -= ROTATION_ANGLE
        self.rotate_map()

    def rotate_map(self):
        self.current_feature_map = rotate((self.unrotated_map * 255).astype(np.uint8), axes=(2, 1), angle=self.rotation_angle, reshape=False, order=1, mode='nearest')
        self.current_feature_map = self.current_feature_map.astype(float) / 255

        self.onMapChange()

    def onMapChange(self):
        self.current_feature_map_w_taget = self.put_target_agent_on_map(self.current_feature_map)
        self.map_roi = self.get_map_roi()

        self.visualization()


    def np_array_to_pixmap(self, arr):
        w, h, ch = arr.shape
        qimg = QtGui.QImage(arr.data.tobytes(), h, w, 3 * h, QtGui.QImage.Format_RGB888)
        qpixmap = QtGui.QPixmap(qimg)

        return qpixmap

    def feature_map_to_img(self, feature_map):
        np_map = np.zeros_like(feature_map[0])

        np_map = cv2.addWeighted(np_map, 1, feature_map[0], 0.2, 0)
        np_map = cv2.addWeighted(np_map, 1, feature_map[6], 0.5, 0)
        np_map = cv2.addWeighted(np_map, 1, feature_map[13], 0.2, 0)
        np_map = cv2.addWeighted(np_map, 1, feature_map[16], 0.1, 0)

        np_map = cv2.flip(np_map, 0)

        np_map = (np_map * 255).astype(np.uint8)
        np_map = (255 - np_map)

        return np_map

    def visualization(self):
        np_map = self.feature_map_to_img(self.current_feature_map)
        #np_roi_map = self.feature_map_to_img(self.map_roi)

        np_map = cv2.cvtColor(np_map, cv2.COLOR_GRAY2RGB)
        #np_roi_map = cv2.cvtColor(np_roi_map, cv2.COLOR_GRAY2RGB)

        np_map = cv2.resize(np_map, (512, 512))
        #np_roi_map = cv2.resize(np_roi_map, (256, 256))

        # draw roi on map
        np_map = cv2.rectangle(np_map, (256 - 128, 256 - 128), (256 + 128, 256 + 128), color=(255, 0, 0), thickness=1)
        car_size = (6, 4)
        np_map = cv2.rectangle(np_map, (256 - car_size[0], 256 - car_size[1]), (256 + car_size[0], 256 + car_size[1]), color=(255, 0, 0), thickness=-1)

        qpixmap = self.np_array_to_pixmap(np_map)
        self.canvas.setPixmap(qpixmap)

        #qpixmap_preview = self.np_array_to_pixmap(np_roi_map)
        #self.preview.setPixmap(qpixmap_preview)

        center_y = int(self.ego_features[0].shape[0] / 2)
        center_x = int(self.ego_features[0].shape[1] / 2)

        vel1 = self.ego_features[1, center_y, center_x]
        vel2 = self.ego_features[2, center_y, center_x]
        acc1 = self.ego_features[3, center_y, center_x]
        acc2 = self.ego_features[4, center_y, center_x]
        yaw = self.ego_features[5, center_y, center_x]

        frmt = "{:.2f}"

        self.textEdit_vel1.setText(frmt.format(vel1))
        self.textEdit_vel2.setText(frmt.format(vel2))
        self.textEdit_acc1.setText(frmt.format(acc1))
        self.textEdit_acc2.setText(frmt.format(acc2))
        self.textEdit_yaw.setText(frmt.format(yaw))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
