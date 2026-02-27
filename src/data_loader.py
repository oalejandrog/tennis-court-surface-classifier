from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add data augmentation to prevent overfitting
train_generated = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = train_generated.flow_from_directory(
    'data/train',
    target_size=(300, 300),
    batch_size=32,
    class_mode='sparse'
)

val_generated = ImageDataGenerator(rescale=1.0/255)
val_data = val_generated.flow_from_directory(
    'data/val',
    target_size=(300, 300),
    batch_size=32,
    class_mode='sparse'
)