import tensorflow as tf
import numpy as np
import os
import argparse
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras
from utils import get_image_paths_and_labels, RBCGenerator
from CNN import create_rbc_model
from collections import Counter

# Argument Parser
parser = argparse.ArgumentParser(description='RBC Classification Testing')
parser.add_argument('--data_root', dest='DATA_ROOT', type=str, 
                    default=r'J:\Saarland University\20250721',
                    help='Root directory of dataset')
parser.add_argument('--log_dir', dest='LOG_DIR', type=str, 
                    default=r'J:\Saarland University\20250721\logs\shallow_pre_overf_aug_20250720_155419', 
                    help='Training log directory containing fold subdirectories')
parser.add_argument('--batch_size', dest='BATCH_SIZE', type=int, default=256,
                    help='Testing batch size')
args = parser.parse_args()

# Load config from training
config_path = os.path.join(args.LOG_DIR, 'config.json')
with open(config_path) as f:
    config = json.load(f)
    
CLASS_NAMES = config['CLASS_NAMES']
IMAGE_SIZE = tuple(config['IMAGE_SIZE'])
NUM_CLASSES = config['NUM_CLASSES']
N_CHANNELS = config['N_CHANNELS']

# Load test dataset
TEST_ROOT = os.path.join(args.DATA_ROOT, 'test')
test_image_paths, test_labels = get_image_paths_and_labels(TEST_ROOT)
print(f"Test samples: {len(test_image_paths)}")

# Create test dataset generator
test_dataset = RBCGenerator(
    image_paths=test_image_paths,
    labels=test_labels,
    num_classes=NUM_CLASSES,
    batch_size=args.BATCH_SIZE,
    augmentation=None,
    image_size=IMAGE_SIZE,
    shuffle=False
)

# Metrics setup
loss_object = keras.losses.CategoricalCrossentropy(from_logits=False)
fold_results = []

# Process each fold
for fold in range(1, 6):
    fold_log_dir = os.path.join(args.LOG_DIR, f'fold_{fold}')
    
    # Find latest checkpoint in the fold directory
    ckpt_files = [f for f in os.listdir(fold_log_dir) 
                 if f.startswith('best_model') and f.endswith('.index')]
    
    if not ckpt_files:
        print(f"No checkpoint found for fold {fold}")
        continue
        
    ckpt_prefix = os.path.join(fold_log_dir, ckpt_files[0].replace('.index', ''))
    print(f"\n{'='*50}")
    print(f"Testing Fold {fold}/5")
    print(f"Using checkpoint: {ckpt_prefix}")
    print(f"{'='*50}")
    
    # Create model
    input_shape = (IMAGE_SIZE[1], IMAGE_SIZE[0], N_CHANNELS)
    model = create_rbc_model(input_shape, NUM_CLASSES)
    
    # Create checkpoint and restore weights
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, directory=fold_log_dir, max_to_keep=1)
    
    # Restore the latest checkpoint
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    print(f"Restored weights from checkpoint: {manager.latest_checkpoint}")
    
    # Test evaluation
    test_loss = keras.metrics.Mean(name='test_loss')
    test_accuracy = keras.metrics.CategoricalAccuracy(name='test_accuracy')
    all_true_labels = []
    all_pred_labels = []
    
    for images, labels in test_dataset:
        predictions = model(images, training=False)
        loss = loss_object(labels, predictions)
        
        test_loss(loss)
        test_accuracy(labels, predictions)
        
        # Convert from one-hot to indices
        true_labels = np.argmax(labels, axis=1)
        pred_labels = np.argmax(predictions.numpy(), axis=1)
        all_true_labels.extend(true_labels)
        all_pred_labels.extend(pred_labels)
    
    # Record results
    test_loss_value = test_loss.result().numpy()
    test_acc_value = test_accuracy.result().numpy()
    
    fold_results.append({
        'fold': fold,
        'test_loss': test_loss_value,
        'test_accuracy': test_acc_value
    })
    
    # Print results
    print(f"Fold {fold} Test Results:")
    print(f"Loss: {test_loss_value:.4f}")
    print(f"Accuracy: {test_acc_value:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_true_labels, all_pred_labels)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )
    
    plt.figure(figsize=(10, 8))
    disp.plot(cmap='Blues', values_format='d', xticks_rotation=45)
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.tight_layout()
    cm_path = os.path.join(fold_log_dir, f'test_confusion_matrix.jpg')
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to: {cm_path}")

# Final summary
print("\nTest Results Summary:")
print("Fold\tTest Loss\tTest Accuracy")
for result in fold_results:
    print(f"{result['fold']}\t{result['test_loss']:.4f}\t{result['test_accuracy']:.4f}")

if fold_results:
    mean_loss = np.mean([r['test_loss'] for r in fold_results])
    mean_acc = np.mean([r['test_accuracy'] for r in fold_results])
    print(f"\nMean Test Loss: {mean_loss:.4f}")
    print(f"Mean Test Accuracy: {mean_acc:.4f}")
else:
    print("\nNo valid test results available")

# Save summary
summary_path = os.path.join(args.LOG_DIR, 'test_summary.txt')
with open(summary_path, 'w') as f:
    f.write("Fold\tTest Loss\tTest Accuracy\n")
    for result in fold_results:
        f.write(f"{result['fold']}\t{result['test_loss']:.4f}\t{result['test_accuracy']:.4f}\n")
    
    if fold_results:
        f.write(f"\nMean Test Loss: {mean_loss:.4f}\n")
        f.write(f"Mean Test Accuracy: {mean_acc:.4f}\n")
    
print(f"Saved test summary to: {summary_path}")