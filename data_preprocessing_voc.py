import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# --- Configuration ---
DATA_DIR = "./VOC2028"
PROCESSED_DATA_DIR = "./processed_images"
IMG_SIZE = 128

def parse_xml_for_objects(xml_path):
    """
    Parses a Pascal VOC XML file and extracts object bounding boxes and labels.
    Returns a list of dictionaries, one for each object.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bbox = obj.find('bndbox')
        
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        objects.append({
            'name': class_name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        })
    return objects

def main():
    """
    Main function to process the Pascal VOC dataset.
    """
    annotations_dir = os.path.join(DATA_DIR, 'Annotations')
    images_dir = os.path.join(DATA_DIR, 'JPEGImages')

    # Collect all unique class names from the dataset first
    all_classes = set()
    for xml_file in os.listdir(annotations_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(annotations_dir, xml_file)
            objects = parse_xml_for_objects(xml_path)
            for obj in objects:
                all_classes.add(obj['name'])

    class_names = sorted(list(all_classes))
    if not class_names:
        print("Error: No classes found in the annotations.")
        return
        
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    
    np.save('image_label_mapping.npy', label_encoder.classes_)
    print(f"Label mapping saved for {len(class_names)} classes: {class_names}")

    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    
    all_xml_files = os.listdir(annotations_dir)
    
    for xml_file in tqdm(all_xml_files, desc="Processing images"):
        if xml_file.endswith('.xml'):
            try:
                xml_path = os.path.join(annotations_dir, xml_file)
                image_name = xml_file.replace('.xml', '.jpg')
                image_path = os.path.join(images_dir, image_name)
                
                image = cv2.imread(image_path)
                if image is None:
                    print(f"\nWarning: Could not read image file {image_path}. Skipping.")
                    continue
                
                objects = parse_xml_for_objects(xml_path)
                
                for obj in objects:
                    obj_class_name = obj['name']
                    xmin, ymin, xmax, ymax = obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']
                    
                    # Crop the object from the image
                    cropped_image = image[ymin:ymax, xmin:xmax]
                    if cropped_image.size == 0:
                        continue
                        
                    resized_image = cv2.resize(cropped_image, (IMG_SIZE, IMG_SIZE))
                    normalized_image = resized_image / 255.0
                    
                    obj_label_int = label_encoder.transform([obj_class_name])[0]
                    
                    processed_class_dir = os.path.join(PROCESSED_DATA_DIR, obj_class_name)
                    if not os.path.exists(processed_class_dir):
                        os.makedirs(processed_class_dir)
                        
                    # Save the processed, cropped image with its label
                    output_file_name = f"{image_name.split('.')[0]}_{obj['name']}_{obj['xmin']}.npy"
                    output_path = os.path.join(processed_class_dir, output_file_name)
                    
                    np.save(output_path, {'image': normalized_image, 'label': obj_label_int})

            except Exception as e:
                print(f"\nError processing XML or image: {xml_file}. Error: {e}")
                continue

if __name__ == "__main__":
    main()