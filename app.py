from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import io
import base64
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model('model/modelo_final.keras')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data)).convert('L')
        
        img_array = np.array(image.resize((60, 60)))/255.0
        img_array = np.expand_dims(img_array, axis = (0, - 1))
        
        prediction = model.predict(img_array)
        probability = float(prediction[0][0])
        
        return jsonify({
            'class': 'Elíptica' if probability > .5479 else 'Espiral',
            'confidence': probability if probability > .5479 else 1 - probability,
            'probability': probability
            'threshold_used': .5479
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False)
