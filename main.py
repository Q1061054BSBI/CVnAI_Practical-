import os
import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report
from sklearn.utils import class_weight

import ssl
import certifi

ssl._create_default_https_context = ssl._create_unverified_context 
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

data_dir_list = ['./dataset/test', './dataset/train', './dataset/val']
categories = ['NORMAL', 'PNEUMONIA']

def preprocess_images(df, target_size=(224, 224)):
    processed_images = []

    for index, row in df.iterrows():
        img_path = row['img_path']
        img = Image.open(img_path)
        img = ImageOps.fit(img, target_size, Image.LANCZOS)
        img_array = np.array(img)
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)

        img_array = img_array / 255.0
        processed_images.append(img_array)

    df['processed_img'] = processed_images
    return df

def plot_sample_images(df, plot_title, samples=5):
    plt.figure(figsize=(12, 10))
    plt.suptitle(plot_title, fontsize=18)

    for idx, category in enumerate(categories):
        sample_images = df[df['category'] == category].sample(samples)

        
        for i, row in enumerate(sample_images.iterrows()):
            img_path = row[1]['img_path'] 
            processed_img = row[1]['processed_img']

            plt.subplot(4, samples, i + 1 + idx * 2 * samples) 
            plt.imshow(np.array(Image.open(img_path)), cmap='gray')
            plt.title(f"{category} (Original)")
            plt.axis('off')

            plt.subplot(4, samples, i + 1 + samples + idx * 2 * samples) 
            plt.imshow(processed_img if isinstance(processed_img, np.ndarray) else np.array(processed_img), cmap='gray')
            plt.title(f"{category} (Processed)")
            plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def augment_images(df):
    transform = A.Compose([
        A.Rotate(limit=30),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])

    count_normal = len(df[df['category'] == 'NORMAL'])
    count_pneumonia = len(df[df['category'] == 'PNEUMONIA'])

    if count_normal < count_pneumonia:
        minority_class = 'NORMAL'
    else:
        minority_class = 'PNEUMONIA'

    diff = abs(count_normal - count_pneumonia)
    
    minority_df = df[df['category'] == minority_class].sample(diff, replace=True)
    
    augmented_images = []
    for index, row in minority_df.iterrows():
        img = row['processed_img'] 
        img = np.array(img)      
        augmented_img = transform(image=img)['image']
        augmented_images.append((augmented_img, minority_class))
    
    for aug_img, category in augmented_images:
        df = df._append({'img_path': None, 'category': category, 'processed_img': aug_img}, ignore_index=True)

    return df


def check_balancing(df, categories):
    result = False
    bigger_length = 0
    for cat in categories:
        class_len = len(df[df['category'] == cat])
        if(bigger_length < class_len):
            bigger_length = class_len

    for cat in categories:
        class_len = len(df[df['category'] == cat])
        if(bigger_length > class_len and (class_len / bigger_length) * 100 > 2):
            result = True

    return result

def prepare_data(df):
    X = np.array(df['processed_img'].tolist()) 
    y = np.array(df['category'].apply(lambda x: 1 if x == 'PNEUMONIA' else 0).tolist()) 
    y = to_categorical(y, num_classes=2) 
    return X, y


def build_model(input_shape):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape) 

    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax', kernel_regularizer=l2(0.001))(x)

    model = Model(inputs=base_model.input, outputs=x)
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, class_weight, epochs=10, batch_size=32, callbacks=None, ):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight
    )
    return history

def plot_training_history(history):
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.show()

    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.legend()
    plt.show()


def evaluate_model(model, X_test, y_test):
    results = model.evaluate(X_test, y_test)
    print(f"Test Loss: {results[0]}")
    print(f"Test Accuracy: {results[1]}")

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print(classification_report(y_true, y_pred_classes, target_names=categories))


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
    df = preprocess_images(df)

    sns.countplot(x='category', data=df)
    plt.title('Classes distribution for '+ dir_titles[i])
    plt.show()

    plot_sample_images(df, plot_title='Image samples of '+dir_titles[i])

    if(check_balancing(df, categories=categories) and data_dir_list[i]!='./dataset/test'):
        df = augment_images(df)

        sns.countplot(x='category', data=df)
        plt.title('Balaces classes distribution for '+ dir_titles[i])
        plt.show()
    

    dataframes.append(df)


train_df = dataframes[1] 
val_df = dataframes[2] 
test_df = dataframes[0]

X_train, y_train = prepare_data(train_df)
X_val, y_val = prepare_data(val_df)
X_test, y_test = prepare_data(test_df)


model = build_model(input_shape)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train.argmax(axis=1)), y=y_train.argmax(axis=1))
class_weights = dict(enumerate(class_weights))

history = train_model(model, X_train, y_train, X_val, y_val, callbacks=[early_stopping], class_weight=class_weights)

plot_training_history(history)

evaluate_model(model, X_test, y_test)   
model.save('pneumonia_detection_model.h5')