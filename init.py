from pymodbus.client import ModbusTcpClient
import psycopg2
import json
from datetime import datetime, time, date
import shutil
from bs4 import BeautifulSoup
import zipfile
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression


# Передача данных из ПЛК в SQL
def from_plc_to_sql():
    # Набор параметров
    with open('config.json', 'r') as f:
        config = json.load(f)
    # Настройка подключения к PostgreSQL
    conn = psycopg2.connect(
        dbname=config["dbname"],
        user=config["user"],
        password=config["password"],
        host=config["host"],
        port=config["port"]
    )
    cursor = conn.cursor()

    # Настройка Modbus TCP клиента
    client = ModbusTcpClient(config['IP_PLC'])  # IP-адрес вашего ПЛК
    client.connect()

    parametrs = ['temp_1', 'humidity_1', 'temp_2', 'humidity_2',
                 'flow_soil_hor_1', 'flow_soil_hor_2',
                 'flow_base_hor_1', 'flow_base_hor_2', 'flow_base_hor_3',
                 'flow_varn_hor_1', 'flow_varn_hor_2',
                 'flow_soil_ver_1', 'flow_soil_ver_2', 'flow_soil_ver_3', 'flow_soil_ver_4',
                 'flow_base_ver_1', 'flow_base_ver_2', 'flow_base_ver_3', 'flow_base_ver_4', 'flow_base_ver_5',
                 'flow_base_ver_6', 'flow_base_ver_7', 'flow_base_ver_8',
                 'flow_varn_ver_1', 'flow_varn_ver_2', 'flow_varn_ver_3', 'flow_varn_ver_4']
    sensor_data = {}
    for parametr in parametrs:
        # Чтение данных с ПЛК
        result = client.read_holding_registers(config[parametr], 1, unit=1)  # В config[parametr] указан адрес
        if result.isError():
            # Ошибка чтения данных с ПЛК
            return False
        else:
            # Обработка полученных данных
            value = result.registers[0]
            sensor_data[parametr] = value

    # Запись в PostgreSQL
    columns = ', '.join(sensor_data.keys())
    placeholders = ', '.join(['%s'] * len(sensor_data))
    values = list(sensor_data.values())

    insert_query = f"INSERT INTO sensor_data ({columns}) VALUES ({placeholders})"
    cursor.execute(insert_query, values)

    conn.commit()

    # Закрываем соединения
    client.close()
    conn.close()

    return True


# Передача данных на заслонки
def on_the_damper(deviations, last_time=datetime.combine(datetime.now().date(), time()), flag):
    # deviations – словарь с отклонениями от нормы в долях от стандартного значения, если значение в пределах нормы то 0
    # last_time – время последней работы функции
    # flag – флаг запуска функции
    new_time = datetime.now()
    if flag and sum(deviations.values()) and (new_time - last_time).total_seconds() > 3600:
        with open('config.json', 'r') as f:
            config = json.load(f)
        # Весовой вклад каждого параметра (настройка вручную или эмпирически)
        k_temp = 0.5
        k_humidity = 0.3
        k_flow = 0.2

        # угол на котоорый надо переместить заслонку
        offset_angle = (k_temp * sum([deviations[key] for key in deviations.keys() if 'temp' in key]) +
                        k_humidity * sum([deviations[key] for key in deviations.keys() if 'humidity' in key]) +
                        k_flow * sum([deviations[key] for key in deviations.keys() if 'flow' in key]))

        # Перезапись в ПЛК
        client = ModbusTcpClient(config['IP_PLC'])  # IP-адрес вашего ПЛК
        client.connect()
        result = client.read_holding_registers(config['angle'], 1, unit=1)
        if not result.isError():
            new_angle = result.registers[0] + offset_angle
            client.write_register(config['angle'], new_angle)
            client.close()
        else:
            return last_time

        # Запись события в файл
        with open("events.txt", "a") as file:
            file.write(f'{new_time.strftime("%Y-%m-%d %H:%M:%S")}\t{str(offset_angle)}\n')

        return new_time

    return last_time


# Сохранение в zip  архив
def add_folder_to_zip(zip_filename, folder_name):
    # Создаем или открываем ZIP-архив для записи
    with zipfile.ZipFile(zip_filename, 'a', zipfile.ZIP_DEFLATED) as zipf:
        # Проходим по всем файлам и папкам внутри папки
        for root, dirs, files in os.walk(folder_name):
            for file in files:
                # Создаем полный путь к файлу
                file_path = os.path.join(root, file)
                # Добавляем файл в архив, сохраняем структуру папок
                zipf.write(file_path, os.path.relpath(file_path, folder_name))


def glasses_seal(t=datetime.now()):
    # t – время для которого собирается отчет, нынешнее по умолчанию
    # Набор параметров
    with open('config.json', 'r') as f:
        config = json.load(f)
    # Установим соединение с PostgreSQL
    conn = psycopg2.connect(
        dbname=config["dbname"],
        user=config["user"],
        password=config["password"],
        host=config["host"],
        port=config["port"]
    )
    # Чтобы выгружать словарь
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # SQL-запрос: найти запись, ближайшую по времени
    query = """
        SELECT *
        FROM sensor_data
        ORDER BY ABS(EXTRACT(EPOCH FROM created_at - %s)) ASC
        LIMIT 1;
    """

    cursor.execute(query, (t,))
    closest_row = cursor.fetchone()

    # Закрытие соединения
    cursor.close()
    conn.close()
    # Добавим 2 колонки
    closest_row['temp'] = (closest_row['temp1'] + closest_row['temp2']) / 2
    closest_row['humidity'] = (closest_row['humidity_1'] + closest_row['humidity_2']) / 2

    # Работа с файлами
    # Наполняем временную папку
    shutil.copytree('report template.files', 'temp_copy/report template.files')
    shutil.copy('report template.htm', 'temp_copy/report template.htm')
    # Читаем HTML-файл
    with open('temp_copy/report template.files/sheet001.htm', 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # Пробегаемся по всем текстовым элементам и делаем замену
    for text_element in soup.find_all(string=True):
        for key in closest_row.keys():
            if key in text_element:
                new_text = text_element.replace(key, str(closest_row[key]))
                text_element.replace_with(new_text)

    # Сохраняем результат в новый файл
    with open('temp_copy/report template.files/sheet001.htm', 'w', encoding='utf-8') as f:
        f.write(str(soup))

    add_folder_to_zip(f'archive {t.strftime("%Y-%m-%d %H:%M:%S")}.zip', 'temp_copy')
    shutil.rmtree('temp_copy')  # Удаляем временную папку

    return True


# Использование временных рядов
def using_time_series(df, param, start_time=datetime.datetime(1970, 1, 1), end_time=datetime.now()):
    # df – данные в виде pd.DataFrame
    # param – параметр
    # start_time – начальное время для графика
    # end_time – конечное время для графика
    df_for_span = df[(df.created_at >= start_time) & (df.created_at <= end_time)]
    return plt.plot(df_for_span['created_at'], df_for_span[param])


# Корреляционный анализ
def correlation_analysis(df, correlation_parameters=None):
    # df – данные в виде pd.DataFrame
    # correlation_parameters – список параметров для которых строим матрицу корелляции, если None-то для всех
    if correlation_parameters is None:
        correlation_parameters = df.columns.tolist()
    return df[correlation_parameters].corr()


# Мониторинг параметров
def monitoring_deviations(df):
    # Стандартные параметры (min,max) брать из config.json
    with open('config.json', 'r') as f:
        config = json.load(f)

    last_row = df.sort_values(by='created_at').iloc[-1].to_dict()
    deviations = {}
    for key in last_row.keys():
        min_key, max_key = config[key]
        if last_row[key] < min_key:
            value = last_row[key] - min_key
        elif last_row[key] > max_key:
            value = last_row[key] - max_key
        else:
            value = 0
        deviations[key] = value

    return deviations


# Обучение модели по предсказанию качества окраски, потом будем обновлять её раз в день
class Model_painting():
    def __init__(self, df):
        df_copy = df.copy()
        # Получить информацию по дням о типе, цвете и доле брака (в ней не содержится сегодняшняя информация)
        # Информация содержится в виде доли по дням: green: n1/n, red:n2/n, type1:n3/n...
        quality_data = pd.read_csv('quality_data.csv')
        self.df = self.data_processing(df_copy, quality_data)

    # Обработка данных
    @staticmethod
    def data_processing(df_copy, quality_data):
        # групировка по дате
        df_copy['date'] = df_copy['created_at'].apply(lambda x: x.date())
        df_group = df_copy.drop('created_at').groupby('date').mean().reset_index()

        quality_data['date'] = quality_data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        # соединяем данные
        return df_group.merge(quality_data, on='date').dropna()

    # Обучаем модель
    def model_training(self):
        X = self.df.drop(columns=['date', 'marriage'])
        y = self.df['marriage']
        self.model = LinearRegression()
        # оценка качества
        score = cross_val_score(self.model, X, y, cv=5, scoring='r2').mean()
        self.model.fit(X, y)
        return score

    # Предсказание на основе сегодняшних данных
    def predict(self, df, data):
        df_copy = df[df.created_at >= datetime.combine(date.today(), time(0, 0))].copy()
        quality_data = pd.DataFrame(data)  # data – словарь с типами и цветами и датой
        data_processed = self.data_processing(df_copy, quality_data)

        X_today = data_processed.drop(columns=['date'])
        return self.model.predict(X_today)
