from flask import Flask, request, jsonify
import pickle
import logging
import numpy as np

# Set up logging
logging.basicConfig(filename='logs.log', level=logging.INFO)

# Load the model
with open("fake_news_model.pki", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    title = data.get("title", "")
    content = data.get("content", "")
    combined_text = title + " " + content

    try:
        # Model expects vectorized input, assuming pipeline handles TF-IDF
        prediction = model.predict([combined_text])[0]
        probabilities = model.predict_proba([combined_text])[0]

        fake_score = round(probabilities[0] * 100, 2)
        true_score = round(probabilities[1] * 100, 2)
        result = "FAKE" if prediction == 0 else "TRUE"

        logging.info(f"Predicted: {result}, Fake: {fake_score}, True: {true_score}")

        return jsonify({
            "result": result,
            "fake_score": fake_score,
            "true_score": true_score
        })

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == "__main__":
    app.run(debug=True)
