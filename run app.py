import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, time, date
from pymodbus.client.sync import ModbusTcpClient
import shutil
import zipfile
import os
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np
from init import (
        from_plc_to_sql, glasses_seal, on_the_damper,
        monitoring_deviations, correlation_analysis,
        using_time_series, Model_painting
)
def run_app():
# Загрузка данных из PostgreSQL
    @st.cache_data
    def load_data():
        with open('config.json', 'r') as f:
            config = json.load(f)
        conn = psycopg2.connect(
            dbname=config["dbname"],
            user=config["user"],
            password=config["password"],
            host=config["host"],
            port=config["port"]
        )
        df = pd.read_sql("SELECT * FROM sensor_data ORDER BY created_at DESC", conn)
        conn.close()
        return df

    st.set_page_config(page_title="Система мониторинга", layout="wide")
    st.title("Система мониторинга и управления оборудованием")

    df = load_data()

    # Боковая панель
    st.sidebar.header("Действия")
    if st.sidebar.button("Считать с ПЛК и записать в БД"):
        success = from_plc_to_sql()
        if success:
            st.sidebar.success("Данные успешно записаны")
        else:
            st.sidebar.error("Ошибка при чтении с ПЛК")

    # Таблица данных с фильтрацией
    st.subheader("Таблица данных")
    columns_to_filter = st.multiselect("Выберите столбцы для отображения:", df.columns, default=df.columns.tolist())
    sort_by = st.selectbox("Сортировать по:", df.columns)
    ascending = st.checkbox("По возрастанию", value=True)
    st.dataframe(df[columns_to_filter].sort_values(by=sort_by, ascending=ascending), use_container_width=True)

    # Матрица корреляции
    st.subheader("Матрица корреляции")
    fig_corr, ax = plt.subplots()
    sns.heatmap(correlation_analysis(df), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig_corr)

    # График по выбору пользователя
    st.subheader("Временной график")
    param = st.selectbox("Параметр:", df.columns)
    start_time = st.date_input("Начальная дата", date.today())
    end_time = st.date_input("Конечная дата", date.today())
    fig_series = plt.figure()
    using_time_series(df, param, datetime.combine(start_time, time()), datetime.combine(end_time, time()))
    st.pyplot(fig_series)

    # Отчёт по замеру
    st.subheader("Отчёт")
    selected_time = st.selectbox("Выберите время замера", df["created_at"].astype(str), index=0)
    if st.button("Скачать ZIP-отчёт"):
        success = glasses_seal(datetime.strptime(selected_time, "%Y-%m-%d %H:%M:%S"))
        if success:
            zip_name = f'archive {selected_time}.zip'
            with open(zip_name, "rb") as file:
                st.download_button(label="Скачать архив", data=file, file_name=zip_name)

    # Уведомления об отклонениях
    st.subheader("Отклонения от нормы")
    deviations = monitoring_deviations(df)
    st.json(deviations)

    # Управление заслонкой
    st.subheader("Управление заслонкой")
    auto_control_flag = st.toggle("Автоматическое управление заслонкой", value=False)
    if st.button("Применить отклонения"):
        new_time = on_the_damper(deviations, flag=auto_control_flag)
        st.success(f"Обновлено: {new_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Прогноз брака
    st.subheader("Прогноз процента брака")
    form_data = {
        'date': [datetime.today().strftime('%Y-%m-%d')],
        st.text_input("Тип изделий"): [1],
        st.text_input("Цвета"): [1]
    }
    if st.button("Прогнозировать"):
        model_obj = Model_painting(df)
        score = model_obj.model_training()
        prediction = model_obj.predict(df, form_data)
        st.metric("Прогнозируемый % брака", f"{prediction[0]*100:.2f}%", help=f"Точность модели: R² = {score:.2f}")

    # Журнал заслонки
    st.subheader("Журнал заслонки")
    if os.path.exists("events.txt"):
        with open("events.txt", "r") as f:
            st.text(f.read())
    else:
        st.info("Журнал пока пуст.")

if __name__ == "__main__":
    run_app()
