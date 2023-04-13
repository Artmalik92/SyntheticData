import sys
import os
import re
import tempfile
import shutil
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QCheckBox, QPushButton, QGridLayout, QLineEdit, QLabel,\
    QComboBox, QFileDialog, QTextEdit, QDesktopWidget, QSpinBox
from PyQt5 import QtGui
from plotting_coordinates import nsk1_test
from Synthetic_data import SyntheticData


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Задаем начальную дату, интервал и общее количество дней
        date_regex = re.compile(r'^\d{4}-\d{2}-\d{2}$')  # регулярное выражение для формата даты 'yyyy-mm-dd'
        self.start_date = None
        self.interval = None  # pd.Timedelta(days=1)
        self.num_periods = None
        self.periods_in_year = None
        self.points_amount = None
        # Создаем список дат
        self.date_list = None

        self.temp_file_xyz_name = ''  # Переменные для хранения имен временных файлов
        self.temp_file_blh_name = ''

        self.script_dir = os.path.dirname(os.path.abspath(__file__))   # Определяем расположение директории
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
        self.choose_interval = QComboBox()
        self.choose_interval.addItems(['W', 'D'])

        # Создаем подобие командной строки
        self.command_line = QTextEdit(self)
        self.command_line.setStyleSheet("background-color: black; color: white;")
        self.command_line.setFont(QtGui.QFont("Courier", 12))

        # Label
        self.label1 = QLabel(self)
        self.label1.setText('Num of periods:')
        self.label2 = QLabel(self)
        self.label2.setText('Start date:')
        self.label3 = QLabel(self)
        self.label3.setText('Interval(week/day):')
        self.label4 = QLabel(self)
        self.label4.setText('Num of points:')

        # Создание поля для ввода
        self.textbox = QLineEdit()
        self.textbox.resize(280, 40)
        self.textbox.setText('104')
        self.textbox2 = QLineEdit()
        self.textbox2.resize(280, 40)
        self.textbox2.setText('2022-01-01')

        # SpinBox
        self.points_spinbox = QSpinBox()
        self.points_spinbox.setRange(3, 15)
        self.points_spinbox.setSingleStep(1)
        self.points_spinbox.setValue(10)

        # функция нажатия кнопки №1 (генерация модели)
        def btn1_press():
            # проверяем корректность введенных пользователем настроек
            if self.textbox.text().isdigit():
                self.num_periods = int(self.textbox.text())
                if self.num_periods < 1:
                    self.command_line.append('Неверно указано кол-во наблюдений')
                    return
            else:
                self.command_line.append('Неверно указано кол-во наблюдений')
                return
            self.start_date = self.textbox2.text()
            if not date_regex.match(self.start_date):  # проверяем соответствие формату даты
                self.command_line.append('Некорректный формат даты')
                return

            # Принимаем выбранное пользователем значение интервала измерений
            self.interval = self.choose_interval.currentText()
            if self.interval == 'W':
                self.periods_in_year = 52
            if self.interval == 'D':
                self.periods_in_year = 365

            # Создаем список дат на основе заданных настроек
            self.date_list = pd.date_range(start=self.start_date, periods=self.num_periods, freq=self.interval)

            # Принимаем от пользователя количество пунктов
            self.points_amount = self.points_spinbox.value()

            # Если остались посторонние временные файлы, удаляем их
            files = os.listdir(self.data_dir)
            for file_name in files:
                if file_name.startswith("temp"):
                    os.remove(os.path.join(self.data_dir, file_name))

            # Создаем временной ряд
            try:
                data = SyntheticData.random_points(56.012255, 82.985018, 141.687, 0.5, self.points_amount)
                data_xyz = SyntheticData.my_geodetic2ecef(data)
                Data_interface_xyz = pd.DataFrame(SyntheticData.create_dataframe(data_xyz, self.date_list))
            except MemoryError as e:
                self.command_line.append(f'Memory error: {e}')
            except Exception as e:
                self.command_line.append(f'Exception: {e}')

            # Проверяем отмеченные пользователем чекбоксы
            if self.checkbox1.isChecked():
                SyntheticData.harmonics(Data_interface_xyz, self.date_list, self.periods_in_year)
            if self.checkbox2.isChecked():
                SyntheticData.linear_trend(Data_interface_xyz, self.date_list, self.periods_in_year)
            if self.checkbox3.isChecked():
                SyntheticData.noise(Data_interface_xyz, self.num_periods)
            if self.checkbox4.isChecked():
                SyntheticData.impulse(Data_interface_xyz, self.num_periods)

            # Переводим временной ряд в BLH
            Data_interface_blh = SyntheticData.my_ecef2geodetic(Data_interface_xyz)

            # Сохраняем DataFrame во временный файл
            with tempfile.NamedTemporaryFile(dir=self.data_dir, mode='r', delete=False, prefix='temp_xyz_',
                                             suffix='.csv') as temp_file_xyz:
                Data_interface_xyz.to_csv(temp_file_xyz.name, sep=';', index=False)
                self.temp_file_xyz_name = temp_file_xyz.name  # сохраняем имя временного файла
            with tempfile.NamedTemporaryFile(dir=self.data_dir, mode='r', delete=False, prefix='temp_blh_',
                                             suffix='.csv') as temp_file_blh:
                Data_interface_blh.to_csv(temp_file_blh.name, sep=';', index=False)
                self.temp_file_blh_name = temp_file_blh.name  # сохраняем имя временного файла
            self.command_line.append("Модель сгенерирована.")

        # Кнопка №2 - графики временного ряда (пока только для NSK1)
        def btn2_press():
            try:
                nsk1_test(self.temp_file_xyz_name, 'NSK1')
            except Exception as e:
                self.command_line.append(f'Error: {e}')

        # Кнопка №3 - Карта геодезической сети (пока только на последнюю дату)
        def btn3_press():
            try:
                data = pd.DataFrame(pd.read_csv(self.temp_file_blh_name, delimiter=';'))
                last_date = data['Date'].max()
                df_on_date = data[data['Date'] == last_date]
                SyntheticData.triangulation(df_on_date)
            except Exception as e:
                self.command_line.append(f'Error: {e}')

        # Кнопка №4 - Сохранение файла
        def btn4_press():
            save_format = self.save_format.currentText()  # Получение формата сохранения файла
            try:
                # Диалоговое окно для выбора имени и расположения файла для сохранения
                user_file_path, user_file_filter = QFileDialog.getSaveFileName(caption="Сохранить файл",
                                                                               directory=
                                                                               self.data_dir+"/data_"+save_format,
                                                                               filter='CSV (*.csv)')
                # копирование из временного файла в новый файл
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

        layout.addWidget(self.label1, 0, 1)
        layout.addWidget(self.textbox, 0, 2)
        layout.addWidget(self.label2, 1, 1)
        layout.addWidget(self.textbox2, 1, 2)
        layout.addWidget(self.label3, 2, 1)
        layout.addWidget(self.choose_interval, 2, 2)
        layout.addWidget(self.label4, 3, 1)
        layout.addWidget(self.points_spinbox, 3, 2)

        layout.addWidget(self.checkbox1, 0, 0)
        layout.addWidget(self.checkbox2, 1, 0)
        layout.addWidget(self.checkbox3, 2, 0)
        layout.addWidget(self.checkbox4, 3, 0)

        layout.addWidget(self.button1, 4, 0)
        layout.addWidget(self.button2, 4, 1)
        layout.addWidget(self.button3, 4, 2)

        layout.addWidget(self.save_format, 5, 0)
        layout.addWidget(self.button4, 5, 1)
        layout.addWidget(self.command_line, 6, 0, 1, -1)

        layout.setRowStretch(6, 0)
        layout.setColumnStretch(0, 1)  # устанавливаем политику растяжения
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)

        self.setLayout(layout)

    # Закрытие окна программы
    def closeEvent(self, event):
        files = os.listdir(self.data_dir)
        for file_name in files:
            if file_name.startswith("temp"):
                os.remove(os.path.join(self.data_dir, file_name))  # Удаляем все временные файлы
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.setGeometry(100, 100, 500, 600)  # установка размеров окна
    window.setWindowTitle('Synthetic TimeSeries data app')
    screen = QDesktopWidget().screenGeometry()  # получаем размер экрана
    size = window.geometry()  # получаем размеры окна
    center = (screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2  # вычисляем центр экрана
    window.move(int(center[0]), int(center[1]))  # перемещаем окно в центр экрана
    window.show()
    sys.exit(app.exec_())

