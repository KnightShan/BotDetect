from flask import Flask, render_template, request, jsonify
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

model = None
scaler = None
model_trained = False


def load_and_clean_data(path=None):
    """Load and clean the bot detection data"""
    try:
        if path and os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Loaded data from {path}")
        else:
            print("CSV file not found, using sample data for demonstration")
            df = create_sample_data()
        
        required_columns = ['User ID', 'Retweet Count', 'Mention Count', 'Follower Count', 'Verified', 'Bot Label']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing columns: {missing_columns}. Using sample data.")
            df = create_sample_data()
        else:
            df = df[required_columns]
        
        df = df.dropna()
        
        df = df.drop_duplicates()
        
        for col in ['User ID', 'Retweet Count', 'Mention Count', 'Follower Count', 'Verified', 'Bot Label']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}. Using sample data.")
        return create_sample_data()

def train_model_advanced(df):
    """Enhanced model training with preprocessing"""
    try:
        X = df.drop(columns=['Bot Label'])
        y = df['Bot Label']
        
        print(f"Training on {len(X)} samples with {len(X.columns)} features")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        scaler = StandardScaler()
        feature_cols = [col for col in X.columns if col != 'User ID']
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[feature_cols] = scaler.fit_transform(X_train[feature_cols])
        X_test_scaled[feature_cols] = scaler.transform(X_test[feature_cols])
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"Training completed successfully!")
        print(f"Training Accuracy: {train_acc:.3f}")
        print(f"Testing Accuracy: {test_acc:.3f}")
        
        return model, scaler, train_acc, test_acc
        
    except Exception as e:
        print(f"Error in model training: {e}")
        raise e

def predict_user_advanced(model, scaler, input_data):
    """Enhanced prediction with preprocessing"""
    try:
        input_array = np.asarray(input_data).reshape(1, -1)
        
        feature_names = ['User ID', 'Retweet Count', 'Mention Count', 'Follower Count', 'Verified']
        input_df = pd.DataFrame(input_array, columns=feature_names)
        
        feature_cols = [col for col in feature_names if col != 'User ID']
        input_scaled = input_df.copy()
        input_scaled[feature_cols] = scaler.transform(input_df[feature_cols])
        
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        result = "BOT" if prediction == 1 else "NOT A BOT"
        
        return result, probabilities
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        raise e

def initialize_model():
    """Initialize and train the model if not already done"""
    global model, scaler, model_trained
    
    if model_trained:
        return True
    
    try:
        if os.path.exists('model.pkl') and os.path.exists('scaler.pkl'):
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            model_trained = True
            print("✅ Loaded pre-trained model and scaler")
            return True
        else:
            print("🔄 Training new model...")
            
            data_paths = [
                "data/bot_detection_data.csv",
                "bot_detection_data.csv",
                "../data/bot_detection_data.csv"
            ]
            
            df = None
            for path in data_paths:
                if os.path.exists(path):
                    df = load_and_clean_data(path)
                    break
            
            if df is None:
                df = load_and_clean_data() 
            
            model, scaler, train_acc, test_acc = train_model_advanced(df)
            
            with open('model.pkl', 'wb') as f:
                pickle.dump(model, f)
            with open('scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            
            model_trained = True
            print(f"✅ Model trained and saved! Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
            return True
            
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return False

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests from the web form"""
    try:
        if not initialize_model():
            return jsonify({'error': 'Model initialization failed. Please try again.'}), 500
        
        # Get form data
        user_id = int(request.form['user_id'])
        retweet_count = int(request.form['retweet_count'])
        mention_count = int(request.form['mention_count'])
        follower_count = int(request.form['follower_count'])
        verified = int(request.form['verified'])
        
        # Validate input
        if any(x < 0 for x in [retweet_count, mention_count, follower_count]):
            return jsonify({'error': 'Counts cannot be negative'}), 400
        
        if verified not in [0, 1]:
            return jsonify({'error': 'Verified must be 0 or 1'}), 400
        
        # Prepare input data
        input_data = (user_id, retweet_count, mention_count, follower_count, verified)
        
        # Make prediction
        result, probabilities = predict_user_advanced(model, scaler, input_data)
        
        # Calculate confidence
        confidence = max(probabilities) * 100
        
        response_data = {
            'prediction': result,
            'confidence': f"{confidence:.1f}%",
            'bot_probability': f"{probabilities[1]*100:.1f}%",
            'human_probability': f"{probabilities[0]*100:.1f}%",
            'input_data': {
                'user_id': user_id,
                'retweet_count': retweet_count,
                'mention_count': mention_count,
                'follower_count': follower_count,
                'verified': bool(verified)
            }
        }
        
        print(f"Prediction made: {result} (Confidence: {confidence:.1f}%)")
        return jsonify(response_data)
        
    except ValueError as e:
        return jsonify({'error': 'Please enter valid numeric values'}), 400
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access"""
    try:
        if not initialize_model():
            return jsonify({'error': 'Model initialization failed'}), 500
        
        data = request.json
        
        required_fields = ['user_id', 'retweet_count', 'mention_count', 'follower_count', 'verified']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        input_data = (
            data['user_id'],
            data['retweet_count'],
            data['mention_count'],
            data['follower_count'],
            data['verified']
        )
        
        result, probabilities = predict_user_advanced(model, scaler, input_data)
        
        return jsonify({
            'prediction': result,
            'bot_probability': float(probabilities[1]),
            'human_probability': float(probabilities[0]),
            'confidence': float(max(probabilities))
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model-info')
def model_info():
    """Get information about the current model"""
    try:
        if not model_trained:
            return jsonify({'error': 'Model not trained yet'}), 400
        
        return jsonify({
            'model_type': 'Logistic Regression',
            'features': ['User ID', 'Retweet Count', 'Mention Count', 'Follower Count', 'Verified'],
            'trained': model_trained,
            'preprocessing': 'StandardScaler applied to numerical features'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        model_status = 'trained' if model_trained else 'not_trained'
        return jsonify({
            'status': 'healthy',
            'model_status': model_status,
            'version': '1.0.0'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Bot Detection Flask App...")
    
    os.makedirs('templates', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    print("📁 Directories created/verified")
    
    print("🤖 Initializing machine learning model...")
    initialize_model()
    
    print("🌐 Starting Flask server...")
    print("📍 Access the app at: http://localhost:5000")
    print("📍 API endpoint: http://localhost:5000/api/predict")
    print("📍 Health check: http://localhost:5000/health")
    
    # Run the app
    app.run(
        debug=True, 
        host='0.0.0.0', 
        port=5000,
        threaded=True
    )