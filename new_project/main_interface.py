import sys
import os
import re
import tempfile
import shutil
import pandas as pd
import numpy as np
from PySide2.QtWidgets import QApplication, QWidget, QCheckBox, QPushButton, QGridLayout, QLineEdit, QLabel,\
    QComboBox, QFileDialog, QTextEdit, QSpinBox
from PySide2 import QtGui
from Synthetic_data import SyntheticData
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        date_regex = re.compile(r'^\d{4}-\d{2}-\d{2}$')  # выражение для формата даты 'yyyy-mm-dd'
        self.start_date = None          # начальная дата
        self.interval = None            # интервал наблюдений
        self.num_periods = None         # кол-во периодов наблюдения (всего)
        self.periods_in_year = None     # кол-во периодов наблюдения в году
        self.points_amount = None       # кол-во пунктов
        self.gen_method = None          # метод генерации
        self.clicked_station = None     # Переменная, отвечающая за событие нажатия на станцию
        self.date_list = None           # список дат
        self.gen_radius = None          # Радиус генерации (относительно стартового пункта)
        self.min_distance = None        # Минимальное расстояние между пунктами
        self.max_distance = None        # Максимальное расстояние между пунктами
        self.impulse_size = None        # Размер импульса (если импульс включен в настройках)
        self.stations_with_impulse_amount = None   # Количество станций с импульсом
        self.impulses_amount = None  # Количество импульсов

        self.temp_file_xyz_name = ''    # Переменные для хранения имен временных файлов
        self.temp_file_blh_name = ''

        self.script_dir = os.path.dirname(os.path.abspath(__file__))   # Определяем расположение директории
        self.data_dir = os.path.join(self.script_dir, "Data")          # Определяем расположения папки для файлов

        self.figure = Figure(figsize=(10, 10), dpi=100)   # Фигура для карты сети
        self.subplot = self.figure.add_subplot(111)

        self.figure2 = Figure(figsize=(10, 10), dpi=100)  # Фигура для временных рядов
        self.axes_1 = self.figure2.add_subplot(3, 1, 1)
        self.axes_2 = self.figure2.add_subplot(3, 1, 2)
        self.axes_3 = self.figure2.add_subplot(3, 1, 3)
        self.figure2.subplots_adjust(hspace=0.5)          # устанавливаем отступы между subplots и фигурой

        self.axes_1.set_xlabel('Interval')
        self.axes_1.set_ylabel('Amplitude')
        self.axes_1.set_title('Coordinate')
        self.axes_2.set_xlabel('Interval')
        self.axes_2.set_ylabel('Amplitude')
        self.axes_2.set_title('Coordinate')
        self.axes_3.set_xlabel('Interval')
        self.axes_3.set_ylabel('Amplitude')
        self.axes_3.set_title('Coordinate')

        self.canvas = FigureCanvas(self.figure)    # подложки
        self.canvas2 = FigureCanvas(self.figure2)

        self.subplot.set_aspect('equal')    # равные стороны графика

        self.canvas.draw()  # рисуем графики
        self.canvas2.draw()

        # Создание кнопок
        self.button1 = QPushButton('Create')
        self.button2 = QPushButton('Show network map')
        self.button3 = QPushButton('Save to file')

        # Создание чекбоксов
        self.checkbox1 = QCheckBox('Harmonics')
        self.checkbox2 = QCheckBox('Linear trend')
        self.checkbox3 = QCheckBox('Noise')
        self.checkbox4 = QCheckBox('Impulse(random)')

        # Создание выпадающих списков
        self.save_format = QComboBox()
        self.save_format.addItems(['XYZ', 'BLH'])
        self.choose_interval = QComboBox()
        self.choose_interval.addItems(['W', 'D'])
        self.choose_method = QComboBox()
        self.choose_method.addItems(['centralized', 'consistent'])
        self.show_format = QComboBox()
        self.show_format.addItems(['XYZ', 'BLH'])

        # Создание командной строки
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
        self.label5 = QLabel(self)
        self.label5.setText('Generation method:')
        self.label6 = QLabel(self)
        self.label6.setText('Coordinate system:')
        self.label7 = QLabel(self)
        self.label7.setText('Gen radius (deg):')
        self.label8 = QLabel(self)
        self.label8.setText('Min distance (m):')
        self.label9 = QLabel(self)
        self.label9.setText('Max distance (m):')
        self.label10 = QLabel(self)
        self.label10.setText('Impulse size (m):')
        self.label11 = QLabel(self)
        self.label11.setText('Stations with impulse:')
        self.label12 = QLabel(self)
        self.label12.setText('Impulses amount:')

        # Создание поля для ввода
        self.textbox = QLineEdit()
        self.textbox.resize(280, 40)
        self.textbox.setText('52')
        self.textbox2 = QLineEdit()
        self.textbox2.resize(280, 40)
        self.textbox2.setText('2022-01-01')
        self.textbox_gen_radius = QLineEdit()
        self.textbox_gen_radius.resize(280, 40)
        self.textbox_gen_radius.setText('0.5')
        self.textbox_min_distance = QLineEdit()
        self.textbox_min_distance.resize(280, 40)
        self.textbox_min_distance.setText('20000')
        self.textbox_max_distance = QLineEdit()
        self.textbox_max_distance.resize(280, 40)
        self.textbox_max_distance.setText('30000')
        self.textbox_impulse_size = QLineEdit()
        self.textbox_impulse_size.resize(280, 40)
        self.textbox_impulse_size.setText('0.05')
        self.textbox_stations_with_impulse_amount = QLineEdit()
        self.textbox_stations_with_impulse_amount.resize(280, 40)
        self.textbox_stations_with_impulse_amount.setText('1')
        self.textbox_impulses_amount = QLineEdit()
        self.textbox_impulses_amount.resize(280, 40)
        self.textbox_impulses_amount.setText('1')

        # SpinBox
        self.points_spinbox = QSpinBox()
        self.points_spinbox.setRange(3, 100)
        self.points_spinbox.setSingleStep(1)
        self.points_spinbox.setValue(10)

        # функция нажатия кнопки №1 (генерация модели)
        def btn1_press():
            self.command_line.append('Создание модели...')
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

            self.points_amount = self.points_spinbox.value()          # Принимаем от пользователя количество пунктов
            self.gen_method = self.choose_method.currentText()        # Принимаем от пользователя метод генерации
            self.gen_radius = float(self.textbox_gen_radius.text())   # Принимаем размер зоны генерации
            self.min_distance = int(self.textbox_min_distance.text())  # Принимаем минимальное
            self.max_distance = int(self.textbox_max_distance.text())  # и максимальное расстояние

            # Если остались посторонние временные файлы, удаляем их
            files = os.listdir(self.data_dir)
            for file_name in files:
                if file_name.startswith("temp"):
                    os.remove(os.path.join(self.data_dir, file_name))

            # Создаем временной ряд
            try:
                data = SyntheticData.random_points(B=56.012255, L=82.985018, H=141.687, zone=self.gen_radius,
                                                   amount=self.points_amount, method=self.gen_method,
                                                   min_dist=self.min_distance, max_dist=self.max_distance)
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
                self.impulse_size = float(self.textbox_impulse_size.text())
                self.stations_with_impulse_amount = int(self.textbox_stations_with_impulse_amount.text())
                self.impulses_amount = int(self.textbox_impulses_amount.text())
                try:
                    dates_list, stations_list = SyntheticData.impulse(df=Data_interface_xyz,
                                                                      impulse_size=self.impulse_size,
                                                                      #target_date='2023-01-01',
                                                                      num_stations=self.stations_with_impulse_amount,
                                                                      random_dates=self.impulses_amount)
                    dates_list_string = '\n'.join(str(item) for item in dates_list)
                    stations_list_string = '\n'.join(str(item) for item in stations_list)
                    self.command_line.append("Выбранные случайные даты для создания импульсов:")
                    self.command_line.append(dates_list_string)
                    self.command_line.append("Выбраны следующие станции для создания импульсов:")
                    self.command_line.append(stations_list_string)
                except Exception as e:
                    print(e)

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

        # Кнопка №2 - Карта геодезической сети (на последнюю дату)
        def btn2_press():
            try:
                data = pd.DataFrame(pd.read_csv(self.temp_file_blh_name, delimiter=';'))
                last_date = data['Date'].max()
                df_on_date = data[data['Date'] == last_date]
                SyntheticData.triangulation(df=df_on_date, subplot=self.subplot, canvas=self.canvas,
                                            max_baseline=self.max_distance)
                # получение изображения графика и отображение на canvas + сохранение в temp
                filename = os.path.join(self.data_dir, "temp.png")
                self.figure.savefig(filename)

            except Exception as e:
                self.command_line.append(f'Error: {e}')

        # Кнопка №3 - Сохранение файла
        def btn3_press():
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

        # Нажатие на станцию (событие)
        def station_click(event):
            try:
                ax = self.subplot
                df = pd.DataFrame(pd.read_csv(self.temp_file_blh_name, delimiter=';'))
                scatter = ax.scatter(df['L'], df['B'])
                if event.button == 1 and scatter.contains(event)[0]:
                    # получаем координаты точки, на которую кликнули
                    x, y = event.xdata, event.ydata
                    # находим ближайшую точку в данных
                    idx = np.sqrt((df['L'] - x) ** 2 + (df['B'] - y) ** 2).idxmin()
                    # получаем название пункта и выводим его на экран
                    station_name = df.loc[idx, 'Station']
                    self.clicked_station = station_name
                    show_format_change()
            except Exception as e:
                self.command_line.append(f'Error: {e}')

        # Изменение формата отображения временных рядов (событие)
        def show_format_change():
            show_format = self.show_format.currentText()
            try:
                if show_format == 'XYZ':
                    df_xyz = pd.DataFrame(pd.read_csv(self.temp_file_xyz_name, delimiter=';'))
                    station_data = df_xyz[df_xyz['Station'] == self.clicked_station]
                    self.axes_1.clear()
                    self.axes_2.clear()
                    self.axes_3.clear()
                    self.axes_1.plot(pd.to_datetime(station_data['Date']), station_data['X'])
                    self.axes_2.plot(pd.to_datetime(station_data['Date']), station_data['Y'])
                    self.axes_3.plot(pd.to_datetime(station_data['Date']), station_data['Z'])
                    self.axes_1.set_xlabel(f'Interval ({self.interval})')
                    self.axes_1.set_ylabel('meters')
                    self.axes_1.set_title('X (ECEF)')
                    self.axes_2.set_xlabel(f'Interval ({self.interval})')
                    self.axes_2.set_ylabel('meters')
                    self.axes_2.set_title('Y (ECEF)')
                    self.axes_3.set_xlabel(f'Interval ({self.interval})')
                    self.axes_3.set_ylabel('meters')
                    self.axes_3.set_title('Z (ECEF)')
                    self.canvas2.draw()
                elif show_format == 'BLH':
                    df_blh = pd.DataFrame(pd.read_csv(self.temp_file_blh_name, delimiter=';'))
                    station_data = df_blh[df_blh['Station'] == self.clicked_station]
                    self.axes_1.clear()
                    self.axes_2.clear()
                    self.axes_3.clear()
                    self.axes_1.plot(pd.to_datetime(station_data['Date']), station_data['B'])
                    self.axes_2.plot(pd.to_datetime(station_data['Date']), station_data['L'])
                    self.axes_3.plot(pd.to_datetime(station_data['Date']), station_data['H'])
                    self.axes_1.set_xlabel(f'Interval ({self.interval})')
                    self.axes_1.set_ylabel('degrees')
                    self.axes_1.set_title('Latitude')
                    self.axes_2.set_xlabel(f'Interval ({self.interval})')
                    self.axes_2.set_ylabel('degrees')
                    self.axes_2.set_title('Longitude')
                    self.axes_3.set_xlabel(f'Interval ({self.interval})')
                    self.axes_3.set_ylabel('meters')
                    self.axes_3.set_title('Height')
                    self.canvas2.draw()
            except Exception as e:
                self.command_line.append(f'Error: {e}')

        self.canvas.mpl_connect('button_press_event', station_click)
        self.show_format.currentIndexChanged.connect(show_format_change)

        # Привязка функций к кнопкам
        self.button1.clicked.connect(btn1_press)
        self.button2.clicked.connect(btn2_press)
        self.button3.clicked.connect(btn3_press)

        # Создание макета и добавление элементов управления
        layout = QGridLayout()

        layout.addWidget(self.label3, 0, 1)            # Interval
        layout.addWidget(self.choose_interval, 0, 2)

        layout.addWidget(self.label1, 1, 1)            # Num of periods
        layout.addWidget(self.textbox, 1, 2)
        layout.addWidget(self.label2, 2, 1)            # Start date
        layout.addWidget(self.textbox2, 2, 2)
        layout.addWidget(self.label4, 3, 1)            # Num of points
        layout.addWidget(self.points_spinbox, 3, 2)
        layout.addWidget(self.label5, 4, 1)            # Method
        layout.addWidget(self.choose_method, 4, 2)
        layout.addWidget(self.label7, 5, 1)                 # Gen radius
        layout.addWidget(self.textbox_gen_radius, 5, 2)
        layout.addWidget(self.label8, 6, 1)
        layout.addWidget(self.textbox_min_distance, 6, 2)   # minimal and maximal distance between points
        layout.addWidget(self.label9, 7, 1)
        layout.addWidget(self.textbox_max_distance, 7, 2)
        layout.addWidget(self.label10, 8, 1)
        layout.addWidget(self.textbox_impulse_size, 8, 2)
        layout.addWidget(self.label11, 9, 1)
        layout.addWidget(self.textbox_stations_with_impulse_amount, 9, 2)
        layout.addWidget(self.label12, 10, 1)
        layout.addWidget(self.textbox_impulses_amount, 10, 2)

        layout.addWidget(self.checkbox1, 0, 0)
        layout.addWidget(self.checkbox2, 1, 0)
        layout.addWidget(self.checkbox3, 2, 0)
        layout.addWidget(self.checkbox4, 3, 0)

        layout.addWidget(self.button1, 11, 0)
        layout.addWidget(self.button2, 11, 1)

        layout.addWidget(self.save_format, 12, 0)
        layout.addWidget(self.button3, 12, 1)

        # Add to the layout at row 7, column 0, spanning 1 row and 3 columns
        layout.addWidget(self.command_line, 13, 0, 1, 3)

        # Виджет для изменения формата отображения временных рядов
        widget = QWidget()
        widget_layout = QGridLayout()
        widget_layout.addWidget(self.label6, 0, 0)
        widget_layout.addWidget(self.show_format, 0, 1)
        widget.setLayout(widget_layout)
        layout.addWidget(widget, 0, 4)

        # Графики matplotlib
        layout.addWidget(self.canvas, 0, 3, 14, 1)
        layout.addWidget(self.canvas2, 1, 4, 14, 1)

        # устанавливаем политику растяжения
        layout.setRowStretch(7, 1)
        layout.setRowStretch(0, 1)
        layout.setColumnStretch(3, 1)
        layout.setColumnStretch(4, 1)
        self.setLayout(layout)

    # Закрытие окна программы
    def closeEvent(self, event):
        files = os.listdir(self.data_dir)
        for file_name in files:
            if file_name.startswith("temp"):
                os.remove(os.path.join(self.data_dir, file_name))  # Удаляем все временные файлы
        event.accept()


if __name__ == '__main__':

    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()

    window = MyWindow()
    window.setWindowTitle('Synthetic TimeSeries data app')
    window.showMaximized()
    sys.exit(app.exec_())

