import os
import numpy as np
import albumentations as A
from PIL import Image
from tensorflow import keras
from imblearn.over_sampling import SMOTE
from collections import Counter

def get_image_paths_and_labels(root_dir):
    """
    Retrieve image paths and labels from directory structure.
    
    Args:
        root_dir: Root directory containing subdirectories named by class labels
        
    Returns:
        tuple: (image_paths, labels) where:
            image_paths: list of full paths to images
            labels: list of integer labels corresponding to directory names
    """
    image_paths = []
    labels = []
    
    # Class to label mapping
    class_mapping = {
        'Ball': 0,
        'Croissant': 1,
        'Pathological': 2,
        'Sheared': 3,
        'Slipper': 4
    }
    
    # Iterate through each class directory
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        # Get label from mapping (default to 5 for unknown classes)
        label = class_mapping.get(class_name, 5)
            
        # Collect all PNG images in the directory
        for file in os.listdir(class_dir):
            if file.lower().endswith('.png'):
                img_path = os.path.join(class_dir, file)
                image_paths.append(img_path)
                labels.append(label)
                
    return image_paths, labels

def load_and_pad_image(img_path, target_size=(200, 168)):
    """
    Load, pad, and preprocess a PNG image to target dimensions.
    
    Args:
        img_path: Path to PNG image file
        target_size: Target dimensions (width, height) to pad to
        
    Returns:
        np.ndarray: Padded image array normalized to [0, 1] with shape (H, W, 1)
    """
    # Load image and convert to grayscale
    img = Image.open(img_path).convert('L')
    
    # Create zero-padded canvas
    padded_img = Image.new('L', target_size, color=0)  # Black background
    
    # Paste original image at top-left corner
    padded_img.paste(img, (0, 0))
    
    # Convert to numpy array
    img_array = np.array(padded_img, dtype=np.float32)
    
    # Normalize to [0, 1]
    if img_array.max() > 1:
        img_array /= 255.0
        
    # Add channel dimension
    img_array = np.expand_dims(img_array, axis=-1)
        
    return img_array

class RBCGenerator(keras.utils.Sequence):
    """
    Data generator for loading and augmenting PNG images.
    
    Args:
        image_paths: List of full paths to PNG images
        labels: Corresponding integer labels
        num_classes: Number of output classes
        batch_size: Size of training batches
        augmentation: Albumentations augmentation pipeline
        shuffle: Whether to shuffle data after each epoch
        image_size: Target image dimensions (width, height)
        n_channels: Number of image channels
    """
    
    def __init__(self, image_paths, labels, num_classes=6, 
                 batch_size=32, augmentation=None, shuffle=True, 
                 image_size=(200, 168), n_channels=1):
        self.image_paths = image_paths
        self.labels = labels
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.shuffle = shuffle
        self.image_size = image_size  # Expected (width, height)
        self.n_channels = n_channels
        self.on_epoch_end()
        
    def __len__(self):
        """Get number of batches per epoch."""
        return int(np.floor(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data."""
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = [self.image_paths[k] for k in indexes]
        batch_labels = [self.labels[k] for k in indexes]
        X, y = self.__data_generation(batch_paths, batch_labels)
        return X, y
    
    def on_epoch_end(self):
        """Update indexes after each epoch."""
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, batch_paths, batch_labels):
        """Generate data for batch."""
        # Initialize arrays with correct dimensions (height, width, channels)
        X = np.empty((self.batch_size, self.image_size[1], self.image_size[0], self.n_channels), 
                     dtype=np.float32)
        y = np.empty((self.batch_size), dtype=int)
        
        for i, path in enumerate(batch_paths):
            # Load and pad image to (200, 168) - (width, height)
            img = load_and_pad_image(path, self.image_size)
            
            # Apply augmentations if specified
            if self.augmentation:
                augmented = self.augmentation(image=img)
                img = augmented["image"]
                
            # Ensure correct shape (H, W, C)
            X[i,] = img
            y[i] = batch_labels[i]
            
        return X, keras.utils.to_categorical(y, num_classes=self.num_classes)


if __name__ == '__main__':
    # Example usage - replace with your actual directory
    root_dir = '/mnt/workspace/test'
    image_paths, labels = get_image_paths_and_labels(root_dir)
    
    # print(labels)

    label_indices = np.array(labels)

    print(type(label_indices[0]))

    # # Define augmentation pipeline
    # augmentation = A.Compose([
    #     A.Rotate(limit=30, p=0.5), 
    #     A.HorizontalFlip(p=0.5),
    #     A.VerticalFlip(p=0.5),
    #     A.RandomBrightnessContrast(p=0.2),
    # ])
    
    # # Initialize generator with target size (200, 168) = (width, height)
    # data_generator = RBCGenerator(
    #     image_paths=image_paths,
    #     labels=labels,
    #     num_classes=6, 
    #     batch_size=512,
    #     augmentation=augmentation,
    #     image_size=(200, 168)  # (width, height)
    # )
    
    # # Test batch generation
    # for image_batch, label_batch in data_generator:
    #     print("Image batch shape:", image_batch.shape)  # Should be (32, 168, 200, 1)
    #     print("Label batch shape:", label_batch.shape)  # Should be (32, 6)
    #     # break  # Only test first batch