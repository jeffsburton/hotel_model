import os
import json
import torch
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import torchvision.transforms as transforms
from classifier import RoomClassifier, get_num_classes_from_checkpoint  # Ensure this is in your project

# Configure paths and constants
BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "casino_floors")
MODEL_PATH = os.path.join(BASE_PATH, "models")
PREDICTIONS = 5

# Create Flask app
app = Flask(__name__, static_folder=os.path.join(BASE_PATH, 'static'),
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))

# load room mapping
with open(MODEL_PATH + "/room_mapping.json", 'r') as json_file:
    mapping = json.load(json_file)

# Load and prepare the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH + "/best_model.pth", map_location=device)
model = RoomClassifier(num_classes=get_num_classes_from_checkpoint(checkpoint))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# Transformation pipeline for images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


@app.route('/')
def index():
    """Serve the test HTML form."""
    return render_template('test.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # Get the image from the request
    image_file = request.files['image']

    try:
        # Load the image and apply transformations
        image = Image.open(image_file).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Get top 5 predictions
            top_probs, top_labels = torch.topk(probabilities, PREDICTIONS)
            top_probs = top_probs.squeeze().cpu().numpy()
            top_labels = top_labels.squeeze().cpu().numpy()

        # Compile results
        results = [
            {'label': int(label), 'confidence': float(prob), 'room_id': mapping[label]}
            for label, prob in zip(top_labels, top_probs)
        ]

        return jsonify({'predictions': results}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Start the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=81)
