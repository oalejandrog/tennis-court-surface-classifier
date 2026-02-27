import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

court_image = image.load_img('test.jpg', target_size(300,300))
court_image_array = image.img_to_array(court_image)
court_image_array = np.expand_dims(court_image_array, axis=0)

model = load_model('tennis_court_model.keras')

prediction = model.predict(court_image_array)

print(prediction)
