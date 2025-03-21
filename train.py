import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import classifier
from classifier import makeParameters



# Base path to save images and JSON
BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "casino_floors")
MODEL_PATH = os.path.join(BASE_PATH, "models")

excluded_rooms = [5, 15, 20, 21, 22, 42, 54, 61, 64, 89, 100, 102, 111, 13, 131, 133, 152, 146, 26, 29, 32, 33, 40, 45, 65, 77, 104, 128, 138, 154]

EPOCHS=30

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# 1. Modify the data transforms to include more augmentation
transform_train = transforms.Compose ( [
    transforms.Resize((256, 256)),  # Larger initial size
    transforms.RandomCrop(224),  # Random crop
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2
    ),
    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



def gatherData(root_path, validation_split=0.1, test_split=.05):
    root_dir = Path(root_path)
    training = []
    validation = []
    test = []
    rooms = {}
    for city_dir in root_dir.iterdir():
        if city_dir.is_dir():
            for hotel_dir in city_dir.iterdir():
                if hotel_dir.is_dir():
                    for room_dir in hotel_dir.iterdir():
                        if room_dir.is_dir():
                            room_id = room_dir.name
                            if int(room_id) in excluded_rooms:
                                continue
                            if room_id not in rooms:
                                rooms[room_id] = []
                            label = len(rooms) - 1
                            images = list(room_dir.glob("*.jpeg"))
                            for img_path in images:
                                sample = {
                                    'path': str(img_path),
                                    'room_id': room_id,
                                    'label': label
                                }
                                rooms[room_id].append(sample)

    for room_id, samples in rooms.items():
        random.shuffle(samples)
        total_length = len(samples)
        split1 = max(1, min(total_length - 2, round(total_length * validation_split)))
        remaining = total_length - split1
        split2 = split1 + max(1, min(remaining - 1, round(total_length * test_split)))
        validation.extend(samples[:split1])
        test.extend(samples[split1:split2])
        training.extend(samples[split2:])

    print(f"Training samples: {len(training)}")
    print(f"Validation samples: {len(validation)}")
    print(f"Test samples: {len(test)}")

    return training, validation, test


class RoomDataset(Dataset):
    def __init__(self, transform=None, samples=[]):
        self.transform = transform
        self.samples = samples
        self.room_to_idx = []
        self.idx_to_room = []
        for sample in self.samples:
            if sample['label'] not in self.room_to_idx:
                idx = len(self.room_to_idx)
                self.room_to_idx.append(sample['label'])
                self.idx_to_room.append(sample['room_id'])
        print(f"Number of unique rooms: {len(self.room_to_idx)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path'])

        if self.transform:
            image = self.transform(image)

        return image, sample['label']


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)




def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scheduler=None):
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    criterion = nn.CrossEntropyLoss()


    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Apply mixup
            images, targets_a, targets_b, lam = mixup_data(images, labels)

            optimizer.zero_grad()
            outputs = model(images)

            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'models/best_model.pth')



        # Update scheduler based on validation accuracy
        scheduler.step(val_acc)

        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
        else:
            patience_counter += 1

        #if patience_counter >= patience:
        #    print(f'Early stopping triggered after epoch {epoch + 1}')
        #   break


        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
        print('-' * 60)

    return train_losses, val_losses, train_accuracies, val_accuracies


def plot_training_progress(train_losses, val_losses, train_accuracies, val_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Losses')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracies')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('models/training_progress.png')
    plt.close()


def predict_room(model, image_path, transform, room_mapping, device):
    model.eval()
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        room_id = room_mapping[predicted.item()]

    return room_id


def main():
    # Create models directory
    os.makedirs('models', exist_ok=True)


    # Create train/val splits
    trainingSamples, validationSamples, testSamples = gatherData(BASE_PATH)


    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_size = 32  # Smaller batch size for better generalization


    # Create datasets with separate transforms
    train_dataset = RoomDataset(transform=transform_train, samples=trainingSamples)
    val_dataset = RoomDataset(transform=transform_val, samples=validationSamples)

    # Create data loaders with appropriate batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model
    num_classes = len(train_dataset.room_to_idx)
    model = classifier.RoomClassifier(num_classes).to(device)

    # Similar parameter groups but with slightly adjusted learning rates


    # Adjusted learning rates for small dataset
    parameters = makeParameters(model)



    optimizer = optim.AdamW(parameters, weight_decay=0.1)  # Increased weight decay

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.7,
        patience=4,
        verbose=True
    )



    # Use weighted cross-entropy loss to handle class imbalance
    class_counts = torch.zeros(num_classes)
    for _, label in train_dataset:
        class_counts[label] += 1
    weights = 1.0 / class_counts
    weights = weights / weights.sum()
    weights = weights.to(device)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=EPOCHS, device=device, scheduler=scheduler
    )

    # Plot and save training progress
    plot_training_progress(train_losses, val_losses, train_accuracies, val_accuracies)

    # Save final model and mappings
    torch.save({
        'model_state_dict': model.state_dict(),
        'room_to_idx': train_dataset.room_to_idx,
        'idx_to_room': train_dataset.idx_to_room,
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }, MODEL_PATH)

    with open(MODEL_PATH, 'w') as f:
        json.dump(testSamples, f, indent=4)


if __name__ == "__main__":
    main()
