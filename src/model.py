from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_tennis_court_surface_classifier():
    classifier = Sequential()

    # The input layer (Updated to 300x300) 
    classifier.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=(300,300,3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # (150x150x32)
    classifier.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    # (75x75x64)
    classifier.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # (37x37x32, padding='valid' by default in MaxPooling2D)
    classifier.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # (18x18x32)
    classifier.add(Flatten())

    # (10,368)
    classifier.add(Dense(units=512, activation='relu'))

    # Dropout
    classifier.add(Dropout(0.5))

    # OUTPUT LAYER: 3 units for our 3 classes (Clay, Grass, Hard Court)
    classifier.add(Dense(units=3, activation='softmax'))

    return classifier

if __name__ == "__main__":
    print("Building the baseline CNN")
    model = build_tennis_court_surface_classifier()
    model.summary()
