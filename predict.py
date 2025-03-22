import json
import os

import torch
from PIL import Image
import torchvision.transforms as transforms
from classifier import RoomClassifier, get_num_classes_from_checkpoint  # assuming this is your classifier class



BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "casino_floors")
MODEL_PATH = os.path.join(BASE_PATH, "models")
PREDICTIONS = 5

def load_test_samples(json_path):
    """Load test samples from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def create_transform():
    """Create transform pipeline for inference"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def main():
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH + '/best_model.pth')
    model = RoomClassifier(num_classes=get_num_classes_from_checkpoint(checkpoint))  # Initialize your classifier
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load test samples
    test_samples = load_test_samples(MODEL_PATH + '/test_samples.json')
    transform = create_transform()

    # Counters for statistics
    top_1_count = 0
    top_2_plus_count = 0
    not_in_top_count = 0
    total_samples = len(test_samples)

    # Process each sample
    with torch.no_grad():
        for sample in test_samples:
            # Load and transform image
            image_path = sample['path']
            true_room_id = sample['room_id']
            label = sample['label']

            try :
                image = Image.open(image_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)

                # Get predictions
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)

                # Get top 5 predictions
                top_probs, top_labels = torch.topk(probabilities, PREDICTIONS)
                top_labels = top_labels.squeeze().cpu().numpy()

                # Check where the true label appears
                if label == top_labels[0]:
                    top_1_count += 1
                elif label in top_labels[1:]:
                    top_2_plus_count += 1
                else:
                    print(f"Image {image_path} from room {true_room_id} not in top {PREDICTIONS}")
                    not_in_top_count += 1

            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue

    # Calculate percentages
    top_1_percent = (top_1_count / total_samples) * 100
    top_2_plus_percent = (top_2_plus_count / total_samples) * 100
    not_in_top_percent = (not_in_top_count / total_samples) * 100

    # Print results
    print("\nResults Summary:")
    print(f"Total samples processed: {total_samples}")
    print(f"Top 1 predictions: {top_1_count} ({top_1_percent:.2f}%)")
    print(f"In top 2-{PREDICTIONS}: {top_2_plus_count} ({top_2_plus_percent:.2f}%)")
    print(f"Not in top {PREDICTIONS}: {not_in_top_count} ({not_in_top_percent:.2f}%)")


if __name__ == "__main__":
    main()
