import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import ssl
import certifi

ssl._create_default_https_context = ssl._create_unverified_context  # Временно отключает проверку сертификатов
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())  # Указывает путь к сертификатам certifi

print(certifi.where())
data_dir_list = ['./dataset/test', './dataset/train', './dataset/val']
categories = ['NORMAL', 'PNEUMONIA']

def plot_sample_images(df, category, samples=5):
    sample_images = df[df['category'] == category].sample(samples)
    plt.figure(figsize=(10, 5))
    for i, row in enumerate(sample_images.iterrows()):
        img = row[1]['processed_img']  # Используем обработанные изображения
        plt.subplot(1, samples, i + 1)
        plt.imshow(img)
        plt.title(category)
        plt.axis('off')
    plt.show()

# Загрузка предобученной модели VGG16
def build_model(input_shape):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)  # Загружаем модель без верхних слоев

    # Замораживаем веса базовой модели
    for layer in base_model.layers:
        layer.trainable = False

    # Добавляем новые полносвязные слои
    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)  # Выходной слой для бинарной классификации (2 класса)

    model = Model(inputs=base_model.input, outputs=x)
    
    # Компилируем модель
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Обучение модели
def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )
    return history

# Оценка модели
def evaluate_model(model, X_test, y_test):
    results = model.evaluate(X_test, y_test)
    print(f"Test Loss: {results[0]}")
    print(f"Test Accuracy: {results[1]}")


def preprocess_images(df, target_size=(224, 224)):
    processed_images = []  # Список для хранения обработанных изображений

    for index, row in df.iterrows():
        img_path = row['img_path']
        category = row['category']

        # Открываем изображение
        img = Image.open(img_path)

        # Изменяем размер изображения
        img_resized = img.resize(target_size)

        # Преобразуем изображение в массив
        img_array = np.array(img_resized)

        # Если изображение черно-белое, преобразуем в RGB
        if len(img_array.shape) == 2:  # Если только 2 измерения (черно-белое изображение)
            img_array = np.stack([img_array] * 3, axis=-1)  # Преобразуем в 3-канальное изображение

        # Нормализуем изображение, приводим значения пикселей к диапазону [0, 1]
        img_array = img_array / 255.0

        # Сохраняем обработанное изображение
        processed_images.append(img_array)

    # Создаем новый столбец с обработанными изображениями
    df['processed_img'] = processed_images

    return df

def prepare_data(df):
    X = np.array(df['processed_img'].tolist())  # Преобразуем изображения в массивы
    y = np.array(df['category'].apply(lambda x: 1 if x == 'PNEUMONIA' else 0).tolist())  # Метки категорий (1 для пневмонии, 0 для нормальных)
    y = to_categorical(y, num_classes=2)  # Преобразуем метки в категориальный формат (для бинарной классификации)
    return X, y


def augment_images(df):
    transform = A.Compose([
        A.Rotate(limit=15),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])

    # Определяем количество изображений для каждого класса
    count_normal = len(df[df['category'] == 'NORMAL'])
    count_pneumonia = len(df[df['category'] == 'PNEUMONIA'])

    # Определяем, какой класс нужно дополнить
    if count_normal < count_pneumonia:
        minority_class = 'NORMAL'
    else:
        minority_class = 'PNEUMONIA'

    # Вычисляем, сколько дополнительных изображений нужно
    diff = abs(count_normal - count_pneumonia)
    
    # Выбираем случайные изображения из меньшинства для аугментации
    minority_df = df[df['category'] == minority_class].sample(diff, replace=True)
    
    augmented_images = []
    for index, row in minority_df.iterrows():
        img = row['processed_img']  # Берем уже обработанное изображение
        
        # Преобразуем изображение обратно в формат numpy для применения аугментации
        img = np.array(img)
        
        # Применяем аугментацию
        augmented_img = transform(image=img)['image']
        
        # Сохраняем аугментированные изображения
        augmented_images.append((augmented_img, minority_class))
    
    # Добавляем новые аугментированные изображения к DataFrame
    for aug_img, category in augmented_images:
        df = df._append({'img_path': None, 'category': category, 'processed_img': aug_img}, ignore_index=True)

    return df


def check_balancing(df, categories):
    result = False
    bigger_length = 0
    for cat in categories:
        print(cat)
        class_len = len(df[df['category'] == cat])
        if(bigger_length < class_len):
            bigger_length = class_len

    for cat in categories:
        class_len = len(df[df['category'] == cat])
        if(bigger_length > class_len and (class_len / bigger_length) * 100 > 2):
            result = True

    return result


dir_titles = ['Test set', 'Train set', 'Configurate (val) set']
dataframes = []
input_shape = (224, 224, 3) 
for i in range(len(data_dir_list)):
    data = []
    for category in categories:
        category_path = os.path.join(data_dir_list[i], category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            data.append((img_path, category))

    df = pd.DataFrame(data, columns=['img_path', 'category'])
    print(df.head())
    # dataframes.append(df)
    df = preprocess_images(df)

    sns.countplot(x='category', data=df)
    plt.title('Classes distribution for '+ dir_titles[i])
    plt.show()

    plot_sample_images(df, category=categories[0])
    plot_sample_images(df, category=categories[1])

    if(check_balancing(df, categories=categories)):
        df = augment_images(df)

        sns.countplot(x='category', data=df)
        plt.title('Balaces classes distribution for '+ dir_titles[i])
        plt.show()
    
    dataframes.append(df)


# Объединяем данные из всех наборов (train, val, test)
final_df = pd.concat(dataframes, ignore_index=True)

# Подготовка данных для обучения
X, y = prepare_data(final_df)

# Разделение на тренировочный, валидационный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Создание и обучение модели
model = build_model(input_shape)
# overfilling on Ege 4
history = train_model(model, X_train, y_train, X_val, y_val, epochs=10)

# Оценка модели на тестовом наборе
evaluate_model(model, X_test, y_test)