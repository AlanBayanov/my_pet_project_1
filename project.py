import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Отключаем лишние логи TensorFlow
import ssl
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Временное отключение SSL-верификации (только для загрузки данных)
ssl._create_default_https_context = ssl._create_unverified_context

try:
    # Попытка загрузить данные стандартным способом
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
except Exception as e:
    print(f"Ошибка при загрузке данных: {e}")
    print("Пробуем альтернативный метод...")
    
    try:
        # Альтернативный способ загрузки
        import urllib.request
        url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
        filename = 'mnist.npz'
        
        urllib.request.urlretrieve(url, filename)
        with np.load(filename, allow_pickle=True) as f:
            train_images, train_labels = f['x_train'], f['y_train']
            test_images, test_labels = f['x_test'], f['y_test']
    except Exception as alt_e:
        print(f"Альтернативная загрузка не удалась: {alt_e}")
        exit()

# Нормализация данных
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Создание модели
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Обучение модели
model.fit(train_images, train_labels, epochs=20)

# Сохранение модели
os.makedirs('models', exist_ok=True)
model.save('models/mnist_model.h5')
print("Модель успешно обучена и сохранена!")

# Оценка модели на тестовых данных
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"\nТочность на тестовых данных: {test_acc:.4f}")

# Дополнительная проверка
print("\nПримеры предсказаний:")
for i in range(5):
    sample = test_images[i].reshape(1, 28, 28)
    prediction = model.predict(sample)
    predicted_digit = np.argmax(prediction)
    true_digit = test_labels[i]
    print(f"Изображение {i}: Предсказано {predicted_digit}, Правильно {true_digit}")