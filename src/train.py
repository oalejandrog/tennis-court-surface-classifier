import tensorflow as tf
from model import build_tennis_court_surface_classifier
from data_loader import train_data, val_data

def train_model():
    model = build_tennis_court_surface_classifier()

    model.compile(
            optimizer = 'adam',
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy']
            )

    EPOCHS = 10

    history = model.fit(
            train_data,
            epochs=EPOCHS,
            validation_data=val_data
            )

    model.save('tennis_court_model.keras')

if __name__ == "__main__":
    train_model()
