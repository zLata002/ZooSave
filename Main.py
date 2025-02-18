import sys
import cv2
import numpy as np
import torch
import asyncio
import logging
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import BufferedInputFile
from PIL import Image, ExifTags
from io import BytesIO

# Замените на токен вашего бота
TELEGRAM_TOKEN = '8069108234:AAGw7NdWRFcaX3LOXrg3HPuHoMXpdAcqcAI'

logging.basicConfig(level=logging.INFO)
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# Добавляем локальный путь к репозиторию YOLOv7
local_repo = r'E:\PyCharmProjects\ZooSaveMain\yolov7'
sys.path.insert(0, local_repo)

# Импортируем класс Model из репозитория YOLOv7
try:
    from models.yolo import Model
except ImportError:
    raise ImportError("Не удалось импортировать класс Model из локального репозитория YOLOv7.")

# Разрешаем безопасную десериализацию для класса models.yolo.Model
torch.serialization.add_safe_globals({'models.yolo.Model': Model})

# Загружаем контрольную точку (checkpoint) вручную
checkpoint = torch.load(r'E:\PyCharmProjects\ZooSaveMain\yolov7.pt',
                        map_location=torch.device('cpu'),
                        weights_only=False)

# Загрузка модели YOLOv7 из локальной копии репозитория
model = torch.hub.load(local_repo, 'custom', checkpoint, source='local', force_reload=True)
model.eval()


def preprocess_image(image):
    """
    Функция для предварительной обработки изображения.
    """
    if image is not None and len(image.shape) == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def draw_boxes(image, detections, confidence_threshold=0.5):
    """
    Рисует прямоугольники на изображении по результатам детекции.
    """
    for _, row in detections.iterrows():
        conf = row['confidence']
        if conf >= confidence_threshold:
            x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = row['name']
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            text = f"{label} {conf:.2f}"
            cv2.putText(image, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


def detect_bird(image, confidence_threshold=0.5):
    """
    Обнаружение объектов на изображении с помощью YOLOv7.
    Возвращает:
    - True, если найдена птица.
    - False, если птица не найдена.
    """
    image = preprocess_image(image)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)
    detections = results.pandas().xyxy[0]

    for _, row in detections.iterrows():
        cls = row['name'].lower()
        conf = row['confidence']
        if conf >= confidence_threshold and cls == "bird":
            logging.info(f"Обнаружена птица: {row}")
            return True  # Найдена птица
    return False  # Птица не найдена


@dp.message(F.text == "/start")
async def send_welcome(message: types.Message):
    """
    Отправляет приветственное сообщение с инструкцией.
    """
    welcome_text = (
        "Привет! Я бот для обнаружения птиц на фото. 🐦📸\n\n"
        "📌 *Как пользоваться?*\n"
        "1️⃣ Отправьте мне фотографию.\n"
        "2️⃣ Я проанализирую изображение.\n"
        "3️⃣ Если на фото найдена птица — сообщу вам об этом и покажу результат!\n"
        "4️⃣ Если птицы нет — отправлю вам фото с выделенными другими объектами, если они есть.\n\n"
        "📸 Попробуйте загрузить фото и получите результат!"
    )
    await message.reply(welcome_text, parse_mode="Markdown")


@dp.message(F.content_type == "photo")
async def handle_photo(message: types.Message):
    """
    Обработчик фотографий:
    - Декодирует изображение.
    - Выполняет детекцию с помощью YOLOv7.
    - Рисует bounding boxes вокруг найденных объектов.
    - Отправляет сообщение с результатами и обработанное фото.
    """
    photo = message.photo[-1]
    file_info = await bot.get_file(photo.file_id)
    data_io = await bot.download_file(file_info.file_path)
    data = data_io.getvalue()

    # Декодируем изображение
    np_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        await message.reply("Не удалось обработать изображение.")
        return

    # Выполняем детекцию
    bird_detected = detect_bird(img)

    # Получаем результаты детекции и рисуем bounding boxes
    img_for_detection = cv2.cvtColor(preprocess_image(img), cv2.COLOR_BGR2RGB)
    detection_results = model(img_for_detection).pandas().xyxy[0]
    img_with_boxes = draw_boxes(img.copy(), detection_results)

    # Кодируем изображение для отправки
    success, encoded_image = cv2.imencode('.jpg', img_with_boxes)
    if not success:
        await message.reply("Не удалось подготовить изображение для отправки.")
        return

    # Оборачиваем изображение в BufferedInputFile
    photo_bytes = BufferedInputFile(encoded_image.tobytes(), filename="result.jpg")

    # Отправляем ответ
    if bird_detected:
        await message.reply_photo(photo=photo_bytes, caption="✅ Птица найдена! 🐦")
    else:
        await message.reply_photo(photo=photo_bytes, caption="❌ На фото нет птиц")


async def main():
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())
