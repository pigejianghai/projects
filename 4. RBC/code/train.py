import random
import tensorflow as tf
import datetime
import numpy as np
import os
import json
import argparse
import matplotlib.pyplot as plt
import albumentations as A
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras
from utils import get_image_paths_and_labels, RBCGenerator
import shutil
from CNN import create_rbc_model
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

################################################################################
# Argument Parser
################################################################################
parser = argparse.ArgumentParser(description='RBC Classification Training')

parser.add_argument('--num_class', dest='n_class', type=int, default=6, 
                    help='Number of classes')
parser.add_argument('--gpu_num', dest='GPU_number', type=str, default='0', 
                    help='CUDA_VISIBLE_DEVICES number')
parser.add_argument('--data_root', dest='DATA_ROOT', type=str, 
                    default=r'J:\Saarland University\20250721',
                    help='Root directory of dataset')
parser.add_argument('--batch_size', dest='BATCH_SIZE', type=int, default=128,
                    help='Training batch size')
parser.add_argument('--epochs', dest='EPOCHS', type=int, default=150,
                    help='Number of training epochs')
parser.add_argument('--lr', dest='LR', type=float, default=1e-4,
                   help='Learning rate')
parser.add_argument('--dropout', dest='DROPOUT', type=float, default=0.5,
                   help='Max dropout rate')
parser.add_argument('--patience', dest='PATIENCE', type=int, default=15,
                   help='Early stopping patience')
parser.add_argument('--log_name', dest='LOG_NAME', type=str, default='rbc_train',
                   help='Base name for log directory')

args = parser.parse_args()

################################################################################
# Reproducibility Setup
################################################################################
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_number

# GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

################################################################################
# Main Configuration
################################################################################
# Dataset paths
TRAIN_ROOT = os.path.join(args.DATA_ROOT, 'train')

# Hyperparameters
BATCH_SIZE = args.BATCH_SIZE
EPOCHS = args.EPOCHS
NUM_CLASSES = args.n_class
IMAGE_SIZE = (200, 168)
N_CHANNELS = 1
N_SPLITS = 5
PATIENCE = args.PATIENCE  # Early stopping patience
AUGMENTATION = A.Compose([
    A.Rotate(limit=90, p=0.75), 
    # A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], seed=RANDOM_STATE)

# Get class names from directory structure
CLASS_NAMES = sorted(os.listdir(TRAIN_ROOT))
print(f"Class names: {CLASS_NAMES}")


################################################################################
# Dataset Preparation
################################################################################
print("Loading datasets...")
train_image_paths, train_labels = get_image_paths_and_labels(TRAIN_ROOT)

print(f"Train samples: {len(train_image_paths)}")

# Check initial class distribution
# train_label_indices = np.argmax(train_labels, axis=1)
train_label_indices = np.array(train_labels)
class_counts = Counter(train_label_indices)
print("Initial class distribution:")
for class_idx, count in sorted(class_counts.items()):
    print(f"  Class {CLASS_NAMES[class_idx]}: {count} samples")


################################################################################
# Metrics and Callbacks
################################################################################
loss_object = keras.losses.CategoricalCrossentropy(from_logits=False)

################################################################################
# Training Loop with 5-Fold Cross Validation
################################################################################
skf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)

print("Starting 5-Fold training...")
fold_results = []

# Create main log directory with custom name
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join(args.DATA_ROOT, 'logs', f'{args.LOG_NAME}_{current_time}')
os.makedirs(log_dir, exist_ok=True)

config = {
    'CLASS_NAMES': CLASS_NAMES,
    'IMAGE_SIZE': IMAGE_SIZE,
    'NUM_CLASSES': NUM_CLASSES,
    'N_CHANNELS': N_CHANNELS
}
with open(os.path.join(log_dir, 'config.json'), 'w') as f:
    json.dump(config, f)

for fold, (train_index, val_index) in enumerate(skf.split(train_image_paths, train_label_indices)):
    print(f"\n{'='*50}")
    print(f"Training Fold {fold+1}/{N_SPLITS}")
    print(f"{'='*50}")
    
    # Create fold-specific log directory
    fold_log_dir = os.path.join(log_dir, f'fold_{fold+1}')
    os.makedirs(fold_log_dir, exist_ok=True)
    
    # Create fold datasets
    fold_train_paths = np.array(train_image_paths)[train_index]
    fold_train_labels = np.array(train_labels)[train_index]
    fold_val_paths = np.array(train_image_paths)[val_index]
    fold_val_labels = np.array(train_labels)[val_index]
    
    # Apply RandomOverSampler to balance classes
    print("Applying class-balanced oversampling...")
    # fold_train_label_indices = np.argmax(fold_train_labels, axis=1)
    fold_train_label_indices = np.array(fold_train_labels)
    print(f"Pre-ROP class distribution: {Counter(fold_train_label_indices)}")
    
    # Reshape paths for sampler
    fold_train_paths_2d = fold_train_paths.reshape(-1, 1)
    
    # Apply RandomOverSampler 
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    fold_train_paths_resampled, fold_train_label_indices_resampled = ros.fit_resample(
        fold_train_paths_2d, fold_train_label_indices
    )
    
    # Convert back to 1D array
    fold_train_paths_resampled = fold_train_paths_resampled.ravel()
    
    # Convert labels back to one-hot
    # fold_train_labels_resampled = tf.keras.utils.to_categorical(
    #     fold_train_label_indices_resampled, num_classes=NUM_CLASSES
    # )
    
    print(f"Post-ROP class distribution: {Counter(fold_train_label_indices_resampled)}")
    print(f"Training samples after ROP: {len(fold_train_paths_resampled)}")

    # Enable eager execution for this fold
    tf.config.experimental_run_functions_eagerly(True)
    
    # Create data generators with resampled data
    train_dataset = RBCGenerator(
        image_paths=fold_train_paths_resampled,
        labels=fold_train_label_indices_resampled,
        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        augmentation=AUGMENTATION,
        image_size=IMAGE_SIZE,
        shuffle=True
    )
    
    val_dataset = RBCGenerator(
        image_paths=fold_val_paths,
        labels=fold_val_labels,
        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=False
    )
    
    # Model creation
    input_shape = (IMAGE_SIZE[1], IMAGE_SIZE[0], N_CHANNELS)
    model = create_rbc_model(input_shape, NUM_CLASSES)
    if fold == 0:
        model.summary()
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=args.LR)
    model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])
    
    # Initialize metrics trackers
    train_loss = keras.metrics.Mean(name='train_loss')
    train_accuracy = keras.metrics.CategoricalAccuracy(name='train_accuracy')
    val_loss = keras.metrics.Mean(name='val_loss')
    val_accuracy = keras.metrics.CategoricalAccuracy(name='val_accuracy')
    
    # History storage
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    best_ckpt_path = None
    wait = 0  # For early stopping counter
    stopped_early = False
    
    # Create checkpoint manager
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, 
        directory=fold_log_dir, 
        max_to_keep=1,
        checkpoint_name='best_model'
    )
    
    # Training loop
    for epoch in range(EPOCHS):
        # Reset metrics
        train_loss.reset_state()
        train_accuracy.reset_state()
        val_loss.reset_state()
        val_accuracy.reset_state()

        # Training phase
        for images, labels in train_dataset:
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = loss_object(labels, predictions)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            train_loss(loss)
            train_accuracy(labels, predictions)

        # Validation phase
        for images, labels in val_dataset:
            predictions = model(images, training=False)
            loss = loss_object(labels, predictions)
            
            val_loss(loss)
            val_accuracy(labels, predictions)

        # Record metrics
        epoch_train_loss = train_loss.result().numpy()
        epoch_train_acc = train_accuracy.result().numpy()
        epoch_val_loss = val_loss.result().numpy()
        epoch_val_acc = val_accuracy.result().numpy()
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Check for improvement
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            wait = 0  # Reset patience counter
            
            # Save checkpoint
            best_ckpt_path = ckpt_manager.save()
            print(f"Saved new best model checkpoint: {best_ckpt_path}")
        else:
            wait += 1  # Increment patience counter
        
        # Print metrics
        print(
            f"Fold {fold+1}/{N_SPLITS} | Epoch {epoch+1:03d}/{EPOCHS} | "
            f"Train: Loss {epoch_train_loss:.4f}, Acc {epoch_train_acc:.4f} | "
            f"Val: Loss {epoch_val_loss:.4f}, Acc {epoch_val_acc:.4f} | "
            f"Best Val Acc: {best_val_acc:.4f} (Epoch {best_epoch+1}) | "
            f"Patience: {wait}/{PATIENCE}"
        )
        
        # Early stopping check
        if wait >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1} - "
                  f"No improvement for {PATIENCE} consecutive epochs")
            stopped_early = True
            break
    
    # Plot and save training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'Fold {fold+1} Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title(f'Fold {fold+1} Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plot_path = os.path.join(fold_log_dir, f'fold_{fold+1}_curves.jpg')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved training curves to: {plot_path}")
    
    # Load best model for test evaluation
    fold_results.append({
        'fold': fold+1,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch+1,
        'stopped_early': stopped_early,
        'final_epoch': epoch+1
    })
    
    print(f"\nFold {fold+1} Training Complete")
    print(f"Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch+1})")
    if stopped_early:
        print(f"Training stopped early at epoch {epoch+1}/{EPOCHS}")

    # Clear session to free memory
    tf.keras.backend.clear_session()
    print('-' * 50)

# Calculate and print final summary
print("\nFinal 5-Fold Cross-Validation Results:")
print("Fold\tBest Val Acc\tBest Epoch\tStopped Early\tFinal Epoch")
for result in fold_results:
    stopped = "Yes" if result['stopped_early'] else "No"
    print(f"{result['fold']}\t{result['best_val_acc']:.4f}\t\t{result['best_epoch']}\t\t{stopped}\t\t{result['final_epoch']}")

# Calculate means excluding failed folds
if fold_results:
    mean_val_acc = np.mean([r['best_val_acc'] for r in fold_results])
else:
    mean_loss = mean_acc = mean_val_acc = float('nan')

print(f"Mean Best Validation Accuracy: {mean_val_acc:.4f}")

# Save summary to file
summary_path = os.path.join(log_dir, 'summary.txt')
with open(summary_path, 'w') as f:
    f.write("Fold\tBest Val Acc\tBest Epoch\tStopped Early\tFinal Epoch\n")
    for result in fold_results:
        stopped = "Yes" if result['stopped_early'] else "No"
        f.write(f"{result['fold']}\t"
                f"{result['best_val_acc']:.4f}\t{result['best_epoch']}\t{stopped}\t{result['final_epoch']}\n")
    
    f.write(f"Mean Best Validation Accuracy: {mean_val_acc:.4f}\n")
    
    early_stop_count = sum(1 for r in fold_results if r['stopped_early'])
    f.write(f"\nEarly stopping occurred in {early_stop_count}/{N_SPLITS} folds\n")

print(f"Saved training summary to: {summary_path}")
print("Training completed!")