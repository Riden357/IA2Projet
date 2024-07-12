import cv2, os
from descriptor import glcm, bitdesc
import numpy as np


def extract_features(image_path, descriptor):
    img = cv2.imread(image_path, 0)
    if img is not None:
        features = descriptor(img)
        return features
    else:
        pass
descriptors = [glcm, bitdesc]
def process_datasets(root_folder):
    
    all_features = [] # List to store all features and metadatas
    for root, dirs, files in os.walk(root_folder):
        #print(root)
        for file in files:
            #print(file)
            if file.lower().endswith(('.jpg','.png', '.jpeg')):
                # Construct relative path
                relative_path = os.path.relpath(os.path.join(root, file), root_folder)
                file_name = f'{relative_path.split("/")[0]}_{file}'
                image_rel_path = os.path.join(root, file)
                folder_name = os.path.basename(os.path.dirname(image_rel_path))
                features = glcm(image_rel_path)
                features = features + [folder_name, relative_path]
                all_features.append(features)
    print(all_features)
    signatures = np.array(all_features)
    np.save('signatures.npy', signatures)
    print('Successfully stored!')
process_datasets('./image')