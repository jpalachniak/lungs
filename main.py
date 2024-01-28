from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pandas as pd
from glob import glob
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

### Kod do sprawdzania czy GPU jest wykryte.
if tf.config.list_physical_devices('GPU'):
    print("GPU is used")
else:
    print("GPU is NOT used")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
### Kod do sprawdzania czy GPU jest wykryte.


# Wczytywanie danych

df = pd.read_csv('../Data_Entry_2017.csv')

# Przetwarzanie etykiet
df_labels = df['Finding Labels'].str.get_dummies(sep='|')
df = pd.concat([df, df_labels], axis=1)

# Znalezienie wszystkich obrazów i ich ścieżek
all_image_paths = {os.path.basename(x): x for x in
                   glob('../images_*/images/*.png')}
print(f"Found {len(all_image_paths)} images in total.")

# Mapowanie nazw obrazów do ich ścieżek w DataFrame
df['image_path'] = df['Image Index'].map(all_image_paths)

# Usunięcie wierszy, gdzie ścieżka jest NaN (czyli pliku nie znaleziono)
df = df.dropna(subset=['image_path'])

print(f"Found {df.shape[0]} images after cleaning.")



# Sprawdzenie ścieżek
for path in df['image_path'].sample(5):
    if not os.path.isfile(path):
        print(f"Brak pliku: {path}")

# Podział na zestaw treningowy i testowy
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Informacje o danych
print("Klasy:", df_labels.columns.tolist())
print("Ilość klas:", len(df_labels.columns))
print("Ilość obrazów w zestawie treningowym:", len(train_df))
print("Ilość obrazów w zestawie testowym:", len(test_df))

# Sprawdzenie kilku ścieżek
print("Sample paths after cleaning:")
print(df['image_path'].sample(5).tolist())

# Generatory obrazów
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_path',
    y_col=df_labels.columns.tolist(),
    class_mode='raw',
    target_size=(128, 128),
    batch_size=64
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='image_path',
    y_col=df_labels.columns.tolist(),
    class_mode='raw',
    target_size=(128, 128),
    batch_size=32
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(df_labels.columns), activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)



history = model.fit(
    train_generator,
    epochs=50,  # Duża liczba epok, early stopping może wcześniej zakończyć trening
    validation_data=test_generator,
    steps_per_epoch=len(train_df) // 32,  # Dostosuj do wielkości twojego batcha
    validation_steps=len(test_df) // 32,  # Dostosuj do wielkości twojego batcha
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

evaluation = model.evaluate(test_generator)
print('Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(evaluation[0], evaluation[1]))


import matplotlib.pyplot as plt
# Po treningu, zapisz model do pliku
model.save('my_model.h5')

# Eksport statystyk uczenia do pliku CSV
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv')

# Generowanie wykresów
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.savefig('training_validation_curves.png')
plt.show()

# Zapisanie wyników ewaluacji
with open('evaluation_results.txt', 'w') as f:
    f.write('Test Loss: {:.4f}, Test Accuracy: {:.4f}\n'.format(evaluation[0], evaluation[1]))

# Jeśli chcesz również wygenerować raport klasyfikacji i macierz pomyłek, możesz użyć:
from sklearn.metrics import classification_report, confusion_matrix

# Generuj przewidywania dla danych testowych
test_steps = len(test_df) // 32
predictions = model.predict(test_generator, steps=test_steps)

# Możesz chcieć przekonwertować przewidywania i prawdziwe etykiety z one-hot encoding na etykiety klas
predicted_classes = predictions.argmax(axis=-1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

conf_matrix = confusion_matrix(true_classes, predicted_classes)
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Zapisz raport klasyfikacji do pliku
with open('classification_report.txt', 'w') as f:
    f.write(classification_report(true_classes, predicted_classes, target_names=class_labels))