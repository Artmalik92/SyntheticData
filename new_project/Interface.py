import sys
import os
import tempfile
import shutil
import uuid
import hashlib
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QCheckBox, QPushButton, QGridLayout, QLineEdit, QLabel,\
    QComboBox, QFileDialog, QTextEdit, QDesktopWidget
from PyQt5 import QtGui
from timeseries_data_new import nsk1_test
from DataFrame_with_coordinates import SyntheticData, date_list, start_date

global num_periods


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.temp_file_xyz_name = ''  # Переменные для хранения имен временных файлов
        self.temp_file_blh_name = ''

        self.script_dir = os.path.dirname(os.path.abspath(__file__))  # Определяем расположение директории
        self.data_dir = os.path.join(self.script_dir, "Data")          # Определяем расположения папки для файлов

        # Создание кнопок
        self.button1 = QPushButton('Create')
        self.button2 = QPushButton('Show plot')
        self.button3 = QPushButton('Show network map')
        self.button4 = QPushButton('Save to file')

        # Создание чекбоксов
        self.checkbox1 = QCheckBox('Harmonics')
        self.checkbox2 = QCheckBox('Linear trend')
        self.checkbox3 = QCheckBox('Noise')
        self.checkbox4 = QCheckBox('Impulse(random)')

        # Создание выпадающего списка с выбором формата сохранения файла
        self.save_format = QComboBox()
        self.save_format.addItems(['XYZ', 'BLH'])

        # Создаем подобие командной строки
        self.command_line = QTextEdit(self)
        self.command_line.setStyleSheet("background-color: black; color: white;")
        self.command_line.setFont(QtGui.QFont("Courier", 12))

        '''
        # Label
        self.label1 = QLabel(self)
        self.label1.setText('Number of periods (weeks)')
        
        # Создание поля для ввода
        self.textbox = QLineEdit()
        self.textbox.resize(280, 40)
        num_periods = self.textbox.text()
        '''
        # Определение функций обработки нажатия кнопок
        def btn1_press():
            files = os.listdir(self.data_dir)
            for file_name in files:
                if file_name.startswith("temp"):
                    os.remove(os.path.join(self.data_dir, file_name))
            try:
                data = SyntheticData.random_points(56.012255, 82.985018, 141.687, 0.5, 10)
                data_xyz = SyntheticData.my_geodetic2ecef(data)
                Data_interface_xyz = pd.DataFrame(SyntheticData.create_dataframe(data_xyz, date_list))
            except MemoryError as e:
                self.command_line.append(f'Memory error: {e}')
            except Exception as e:
                self.command_line.append(f'Exception: {e}')

            if self.checkbox1.isChecked():
                SyntheticData.harmonics(Data_interface_xyz, date_list)
            if self.checkbox2.isChecked():
                SyntheticData.linear_trend(Data_interface_xyz, date_list)
            if self.checkbox3.isChecked():
                SyntheticData.noise(Data_interface_xyz)
            if self.checkbox4.isChecked():
                SyntheticData.impulse(Data_interface_xyz)

            Data_interface_blh = SyntheticData.my_ecef2geodetic(Data_interface_xyz)
            # Сохраняем DataFrame в файл
            '''
            file_id = str(uuid.uuid1().hex) # Уникальный идентификатор uuid.
            file_id = hashlib.sha1(os.urandom(128)).hexdigest()[:4]
            '''

            with tempfile.NamedTemporaryFile(dir=self.data_dir, mode='r', delete=False, prefix='temp_xyz_',
                                             suffix='.csv') as temp_file_xyz:
                # Сохраняем DataFrame в файл XYZ
                Data_interface_xyz.to_csv(temp_file_xyz.name, sep=';', index=False)
                self.temp_file_xyz_name = temp_file_xyz.name  # сохраняем имя временного файла

            with tempfile.NamedTemporaryFile(dir=self.data_dir, mode='r', delete=False, prefix='temp_blh_',
                                             suffix='.csv') as temp_file_blh:
                # Сохраняем DataFrame в файл BLH
                Data_interface_blh.to_csv(temp_file_blh.name, sep=';', index=False)
                self.temp_file_blh_name = temp_file_blh.name  # сохраняем имя временного файла

            self.command_line.append("Модель сгенерирована.")

        def btn2_press(filename_XYZ):
            try:
                nsk1_test(f'Data/{filename_XYZ}', 'NSK1')
            except Exception as e:
                self.command_line.append(f'Error: {e}')

        def btn3_press():
            try:
                data = pd.DataFrame(pd.read_csv('data.csv', delimiter=';'))
                data_blh = SyntheticData.my_ecef2geodetic(data)
                last_date = data_blh['Date'].max()
                df_on_date = data_blh[data_blh['Date'] == last_date]
                SyntheticData.triangulation(df_on_date)
                print(df_on_date)
            except Exception as e:
                self.command_line.append(f'Error: {e}')

        def btn4_press():
            save_format = self.save_format.currentText() # Получение формата сохранения файла
            try:
                # Откройте диалоговое окно для выбора имени и расположения файла для сохранения
                user_file_path, user_file_filter = QFileDialog.getSaveFileName(caption="Сохранить файл",
                                                                               directory=
                                                                               self.data_dir+"/data_"+save_format,
                                                                               filter='CSV (*.csv)')
                # Если пользователь выбрал файл, скопируйте содержимое временного файла в этот файл
                if user_file_path:
                    with open(user_file_path, "w") as user_file:
                        # Сохраняем DataFrame в файл в выбранном формате
                        if save_format == 'XYZ':
                            shutil.copyfileobj(open(self.temp_file_xyz_name, "r"), user_file)
                            self.command_line.append("Файл сохранен в системе XYZ")
                        elif save_format == 'BLH':
                            shutil.copyfileobj(open(self.temp_file_blh_name, "r"), user_file)
                            self.command_line.append("Файл сохранен в системе BLH")

            except Exception as e:
                self.command_line.append(f'Error: {e}')

        # Привязка функций к кнопкам
        self.button1.clicked.connect(btn1_press)
        self.button2.clicked.connect(btn2_press)
        self.button3.clicked.connect(btn3_press)
        self.button4.clicked.connect(btn4_press)

        # Создание макета и добавление элементов управления
        layout = QGridLayout()
        '''
        layout.addWidget(self.label1, 0, 1)
        layout.addWidget(self.textbox, 1, 1)
        '''
        layout.addWidget(self.checkbox1, 1, 0)
        layout.addWidget(self.checkbox2, 2, 0)
        layout.addWidget(self.checkbox3, 3, 0)
        layout.addWidget(self.checkbox4, 4, 0)

        layout.addWidget(self.button1, 5, 0)
        layout.addWidget(self.button2, 5, 1)
        layout.addWidget(self.button3, 5, 2)

        layout.addWidget(self.save_format, 6, 0)
        layout.addWidget(self.button4, 6, 1)
        layout.addWidget(self.command_line, 7, 0, 1, -1)
        layout.setRowStretch(7, 0)  # устанавливаем политику растяжения для 8-й строки
        layout.setColumnStretch(0, 1)  # устанавливаем политику растяжения
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)

        self.setLayout(layout)

    def closeEvent(self, event):
        # Удаляем все временные файлы
        files = os.listdir(self.data_dir)
        for file_name in files:
            if file_name.startswith("temp"):
                os.remove(os.path.join(self.data_dir, file_name))
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.setGeometry(100, 100, 300, 600)  # установка размеров окна
    window.setWindowTitle('Synthetic TimeSeries data app')
    screen = QDesktopWidget().screenGeometry()  # получаем размер экрана
    size = window.geometry()  # получаем размеры окна
    center = (screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2  # вычисляем центр экрана
    window.move(int(center[0]), int(center[1]))  # перемещаем окно в центр экрана
    window.show()
    sys.exit(app.exec_())






