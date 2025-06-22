import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

def build_posters_features():
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    posters_dir = "data/posters"
    features = []
    names = []
    for fname in os.listdir(posters_dir):
        fpath = os.path.join(posters_dir, fname)
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = image.load_img(fpath, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feat = model.predict(x).flatten()
            features.append(feat)
            names.append(fname)
    np.save("posters_features.npy", np.array(features))
    np.save("posters_names.npy", np.array(names))

if __name__ == "__main__":
    build_posters_features() 