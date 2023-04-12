import pandas as pd
import tempfile
import os
import shutil
from PyQt5 import QtWidgets

# Создаем DataFrame с какими-то данными
df = pd.DataFrame({'Column1': [1, 2, 3], 'Column2': ['A', 'B', 'C']})

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "Data")

# Создаем временный файл
with tempfile.NamedTemporaryFile(dir=data_dir, mode='r', delete=False, suffix='.csv') as temp_file:
    # Сохраняем DataFrame в файл
    df.to_csv(temp_file.name, sep=';', index=False)

    # Создайте приложение PyQt5
    app = QtWidgets.QApplication([])

    # Откройте диалоговое окно для выбора имени и расположения файла для сохранения
    user_file_path, user_file_filter = QtWidgets.QFileDialog.getSaveFileName(filter='CSV (*.csv)')
    # Если пользователь выбрал файл, скопируйте содержимое временного файла в этот файл
    if user_file_path:
        with open(user_file_path, "w") as user_file:
            shutil.copyfileobj(temp_file, user_file)

    # Закрываем временный файл
    temp_file.close()

    # Удаляем временный файл
    os.remove(temp_file.name)

    # Закройте приложение PyQt5
    app.exit()

