from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import tensorflow_datasets as tfds

app = Flask(__name__)

# Завантаження моделі
model = tf.keras.models.load_model('flowers_classifier.h5')

# Загрузка датасета Oxford Flowers 102
dataset, info = tfds.load('oxford_flowers102', with_info=True)

# Получаем имена классов
class_names = [info.features['label'].int2str(i) for i in range(102)]  # 102 класса в датасете

IMG_SIZE = 128

# Функция для предсказания квітки
def predict_flower(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Додавання розміру партії
    img_array = img_array / 255.0  # Нормалізація

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Перевірка на коректність індексу
    if predicted_class < len(class_names):
        return class_names[predicted_class]
    else:
        return "Unknown class"  # Якщо індекс не в межах класів

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Створюємо шлях до завантаженого файлу
        file_path = os.path.join('uploads', file.filename)
        
        # Перевірка на наявність папки для завантажених файлів
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        # Зберігаємо файл
        file.save(file_path)
        
        # Отримуємо результат предсказання
        flower_class = predict_flower(file_path)
        
        return render_template('result.html', flower_class=flower_class)

if __name__ == "__main__":
    app.run(debug=True)
