import os
import datetime

log_file_path = "logging.txt"

# Функция для записи логов
def log_event(event_type, message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"{timestamp} - {event_type} - {message}\n"
    with open(log_file_path, "a") as log_file:
        log_file.write(log_message)

# Симуляция событий сервера
def simulate_server_events():
    log_event("INFO", "Сервер запущен.")
    log_event("INFO", "Запрос получен от клиента 192.168.0.1.")
    log_event("ERROR", "Ошибка подключения к базе данных.")
    log_event("INFO", "Запрос от клиента 192.168.0.2 обработан успешно.")
    log_event("WARNING", "Низкое количество свободной памяти.")
    log_event("INFO", "Сервер остановлен.")

# Чтение и вывод лог-файла
def read_log_file():
    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as log_file:
            print("Содержимое лог-файла:")
            for line in log_file:
                print(line.strip())
    else:
        print("Лог-файл не найден.")
