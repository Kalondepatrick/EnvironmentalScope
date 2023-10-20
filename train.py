import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the path to your dataset directory.
dataset_dir = 'uploads'

# Define the image dimensions and batch size.
image_height = 224
image_width = 224
batch_size = 32

# Create data generators for training and validation sets.
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1]
    rotation_range=20,    # Data augmentation: random rotation
    width_shift_range=0.2,  # Data augmentation: horizontal shift
    height_shift_range=0.2,  # Data augmentation: vertical shift
    horizontal_flip=True,    # Data augmentation: horizontal flip
    validation_split=0.2  # Split data into training and validation
)

train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Specify training set
)

validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Specify validation set
)

# Create a CNN model.
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')  # Two classes: apple scab and healthy
])

# Compile the model.
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model.
num_epochs = 10
history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=validation_generator
)

# Evaluate the model on the test set.
test_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy}')

# Save the trained model as a .h5 file.
model.save('crop_disease_model.h5')
print('Model saved as crop_disease_model.h5')