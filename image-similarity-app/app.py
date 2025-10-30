import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from model_utils import ImageSimilarityModel
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = ImageSimilarityModel()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'jfif', 'pjpeg', 'pjp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app.json_encoder = NumpyEncoder

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_similarity():
    try:
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': 'Please upload two images'}), 400
        
        image1 = request.files['image1']
        image2 = request.files['image2']
        
        if image1.filename == '' or image2.filename == '':
            return jsonify({'error': 'Please select two images'}), 400
        
        if not allowed_file(image1.filename) or not allowed_file(image2.filename):
            return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        filename1 = secure_filename(image1.filename)
        filename2 = secure_filename(image2.filename)
        
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        
        image1.save(filepath1)
        image2.save(filepath2)
        
        results = model.analyze_similarity(filepath1, filepath2)
        
        # Cleanup
        try:
            os.remove(filepath1)
            os.remove(filepath2)
        except:
            pass
        
        return jsonify(results)
            
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/demo/<scenario>')
def demo_scenario(scenario):
    try:
        demo_results = {
            'same-object': {
                'overall_similarity': 85.5,
                'breakdown': {'object_similarity': 92.3, 'texture_similarity': 78.1, 'color_similarity': 65.7},
                'layer_contributions': {'low_level': 65.7, 'mid_level': 78.1, 'high_level': 92.3},
                'detected_objects': [{'name': 'cat', 'confidence': 92}, {'name': 'couch', 'confidence': 85}]
            },
            'similar-texture': {
                'overall_similarity': 72.3,
                'breakdown': {'object_similarity': 45.2, 'texture_similarity': 88.9, 'color_similarity': 67.8},
                'layer_contributions': {'low_level': 88.9, 'mid_level': 67.8, 'high_level': 45.2},
                'detected_objects': [{'name': 'fabric', 'confidence': 82}, {'name': 'textile', 'confidence': 78}]
            },
            'product-discovery': {
                'overall_similarity': 68.9,
                'breakdown': {'object_similarity': 75.4, 'texture_similarity': 62.1, 'color_similarity': 59.8},
                'layer_contributions': {'low_level': 59.8, 'mid_level': 62.1, 'high_level': 75.4},
                'detected_objects': [{'name': 'shoe', 'confidence': 88}, {'name': 'sneaker', 'confidence': 79}]
            },
            'duplicate-detection': {
                'overall_similarity': 95.2,
                'breakdown': {'object_similarity': 96.8, 'texture_similarity': 93.4, 'color_similarity': 91.7},
                'layer_contributions': {'low_level': 91.7, 'mid_level': 93.4, 'high_level': 96.8},
                'detected_objects': [{'name': 'document', 'confidence': 94}, {'name': 'paper', 'confidence': 89}]
            }
        }
        
        if scenario in demo_results:
            results = demo_results[scenario]
            results['success'] = True
            results['processing_time'] = f"{np.random.uniform(1, 3):.1f}s"
            return jsonify(results)
        
        return jsonify({'error': 'Demo scenario not available'}), 404
        
    except Exception as e:
        return jsonify({'error': f'Demo error: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Image Similarity Analysis Server...")
    print("Object detection model initialized!")
    app.run(debug=True, host='0.0.0.0', port=5000)