import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from tensorflow import keras

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Распознавание рукописных цифр")
        
        # Загрузка обученной модели
        self.model = keras.models.load_model('models/mnist_model.h5')
        
        # Создаем область для рисования
        self.canvas = tk.Canvas(root, width=280, height=280, bg="black")
        self.canvas.pack()
        
        # Кнопки
        self.btn_recognize = tk.Button(root, text="Распознать", command=self.recognize_digit)
        self.btn_recognize.pack(side=tk.LEFT)
        
        self.btn_clear = tk.Button(root, text="Очистить", command=self.clear_canvas)
        self.btn_clear.pack(side=tk.RIGHT)
        
        # Настройки рисования
        self.image = Image.new("L", (280, 280), 0)  # Черный фон
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)
    
    def paint(self, event):
        """Рисование на холсте"""
        x, y = event.x, event.y
        r = 15  # Увеличим толщину кисти
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=255)  # Белая цифра
    
    def clear_canvas(self):
        """Очистка холста"""
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
    
    def recognize_digit(self):
        """Распознавание цифры"""
        try:
            # Подготовка изображения (КРИТИЧНО ВАЖНЫЙ БЛОК)
            img = self.image.resize((28, 28))
            img = ImageOps.invert(img)  # Инверсия цветов (MNIST использует белые цифры на черном)
            img_array = np.array(img)
            
            # Нормализация и добавление размерности
            img_array = img_array.astype('float32') / 255.0
            img_array = img_array.reshape(1, 28, 28)  # Добавляем размерность батча
            
            # Распознавание
            prediction = self.model.predict(img_array)
            digit = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # Показываем результат
            messagebox.showinfo(
                "Результат",
                f"Я думаю, это цифра: {digit}\nУверенность: {confidence*100:.1f}%"
            )
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось распознать цифру: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()