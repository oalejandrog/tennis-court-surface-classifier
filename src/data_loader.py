from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_generated = ImageDataGenerator(rescale=1.0/255)

train_data = train_generated.flow_from_directory(
    'data/train',
    target_size=(300, 300),
    batch_size=32,
    class_mode='sparse'
)