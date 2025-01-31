from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import yaml
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediciton import PredictionPipeline


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


from ruamel.yaml import YAML

def update_yaml(file_path, new_data):
    yaml = YAML()
    yaml.preserve_quotes = True

    # Read the existing YAML file
    with open(file_path, "r") as f:
        yaml_data = yaml.load(f)

    # Update nested keys correctly
    for key, value in new_data.items():
        if isinstance(value, dict):
            # If the value is a dictionary, update nested keys
            for nested_key, nested_value in value.items():
                if key in yaml_data:
                    yaml_data[key][nested_key] = nested_value
        else:
            # Update top-level keys
            yaml_data[key] = value

    # Write the updated YAML back to the file
    with open(file_path, "w") as f:
        yaml.dump(yaml_data, f)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    if request.method == 'POST':
        try:
            dataset_url = request.form.get('dataset_url')
            epochs = int(request.form.get('epochs'))
            learning_rate = float(request.form.get('learning_rate'))
            batch_size = int(request.form.get('batch_size'))
            base_model = request.form.get('base_model')  # Get the selected base model

            # Update YAML files
            update_yaml("config/config.yaml", {"data_ingestion": {"source_URL": dataset_url}})
            update_yaml("params.yaml", {
                "EPOCHS": epochs,
                "LEARNING_RATE": learning_rate,
                "BATCH_SIZE": batch_size,
                "IMAGE_SIZE": [224, 224, 3],
                "BASE_MODEL": base_model  # Add the selected base model to params.yaml
            })

            # Trigger training
            os.system("dvc repro")
            print("Training started successfully!", "success")  # Success message
        except Exception as e:
            print(f"An error occurred: {str(e)}", "error")  # Error message

        return redirect(url_for('trainRoute'))

    return render_template('train.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict()
    return jsonify(result)


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=5000)
