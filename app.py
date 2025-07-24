from flask import Flask, render_template, request, jsonify
from flask import session, redirect, url_for, flash
from flask_cors import CORS
import re
import joblib
import os
import logging
import sys
import pandas as pd
from werkzeug.utils import secure_filename
import csv
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  

app.secret_key = 'your_secret_key_123'  

# Upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


USERS = {
    'admin': 'password123',
    'user': 'user123'
}

# Global variables for models
svm_model = None
gnb_model = None
vectorizer = None
stemmer = None
stop_remover = None

def login_required(f):
    """Decorator to require login for certain routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        print(f"ERROR: Missing dependency - {e}")
        print("Please install Sastrawi: pip install Sastrawi")
        return False

def load_models():
    """Load ML models and preprocessing tools"""
    global svm_model, gnb_model, vectorizer, stemmer, stop_remover
    
    try:
        # Check dependencies first
        if not check_dependencies():
            return False
            
        # Import after dependency check
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        
        # Check if model files exist
        model_files = [
            "models/svm_model.pkl",
            "models/gnb_model.pkl", 
            "models/tfidf_vectorizer.pkl"
        ]
        
        for file_path in model_files:
            if not os.path.exists(file_path):
                logger.error(f"Model file not found: {file_path}")
                print(f"ERROR: Model file not found - {file_path}")
                return False
        
        # Load models
        logger.info("Loading SVM model...")
        svm_model = joblib.load("models/svm_model.pkl")
        
        logger.info("Loading GNB model...")
        gnb_model = joblib.load("models/gnb_model.pkl")
        
        logger.info("Loading TF-IDF vectorizer...")
        vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
        
        # Initialize preprocessing tools
        logger.info("Initializing stemmer...")
        stemmer = StemmerFactory().create_stemmer()
        
        logger.info("Initializing stopword remover...")
        stop_remover = StopWordRemoverFactory().create_stop_word_remover()
        
        logger.info("All models and preprocessing tools loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        print(f"ERROR loading models: {str(e)}")
        return False

def preprocess_text(text):
    """Basic text preprocessing"""
    if not text or not isinstance(text, str):
        return ""
    
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)
        
        # Remove punctuation and special characters
        text = re.sub(r"[^\w\s]", "", text)
        
        # Remove numbers
        text = re.sub(r"\d+", "", text)
        
        # Remove extra whitespaces
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    except Exception as e:
        logger.error(f"Error in basic preprocessing: {str(e)}")
        return text

def advanced_clean(text):
    """Advanced text cleaning with stemming and stopword removal"""
    if not text:
        return ""
    
    try:
        # Remove stopwords
        text = stop_remover.remove(text)
        
        # Stemming
        text = stemmer.stem(text)
        
        return text
    except Exception as e:
        logger.error(f"Error in advanced cleaning: {str(e)}")
        return text

def predict_sentiment(text):
    """Predict sentiment using both SVM and GNB models"""
    try:
        if not text or not text.strip():
            return None, None, "Input text is empty"
        
        # Check if models are loaded
        if any(model is None for model in [svm_model, gnb_model, vectorizer, stemmer, stop_remover]):
            return None, None, "Models not loaded properly"
        
        # Preprocessing
        clean = preprocess_text(text)
        if not clean:
            return None, None, "Text became empty after preprocessing"
            
        final = advanced_clean(clean)
        if not final:
            return None, None, "Text became empty after advanced cleaning"
        
        # Transform text to TF-IDF
        tfidf_input = vectorizer.transform([final])
        
        # Make predictions
        svm_pred = svm_model.predict(tfidf_input)[0]
        gnb_pred = gnb_model.predict(tfidf_input.toarray())[0]
        
        # Map predictions to labels
        label_map = {0: "Negatif", 1: "Netral", 2: "Positif"}
        
        return label_map[svm_pred], label_map[gnb_pred], None
        
    except Exception as e:
        error_msg = f"Error in prediction: {str(e)}"
        logger.error(error_msg)
        return None, None, error_msg

def process_csv_file(file_path):
    """Process uploaded CSV file and return predictions"""
    results = []
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Assume the CSV has a column named 'text' or 'review' or similar
        text_column = None
        for col in df.columns:
            if col.lower() in ['text', 'review', 'ulasan', 'komentar', 'comment']:
                text_column = col
                break
        
        if text_column is None:
            # If no standard column found, use the first column
            text_column = df.columns[0]
        
        # Process each row
        for index, row in df.iterrows():
            text = str(row[text_column]).strip()
            if text and text != 'nan':
                svm_result, gnb_result, error = predict_sentiment(text)
                
                if error:
                    svm_result = "Error"
                    gnb_result = "Error"
                
                results.append({
                    'text': text[:100] + ('...' if len(text) > 100 else ''),
                    'svm': svm_result or "Error",
                    'gnb': gnb_result or "Error"
                })
            else:
                results.append({
                    'text': "Empty text",
                    'svm': "Skipped",
                    'gnb': "Skipped"
                })
                
    except Exception as e:
        logger.error(f"Error processing CSV: {str(e)}")
        results.append({
            'text': f"Error reading file: {str(e)}",
            'svm': "Error",
            'gnb': "Error"
        })
    
    return results

# Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in USERS and USERS[username] == password:
            session['logged_in'] = True
            session['username'] = username
            flash('Login berhasil!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Username atau password salah!', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    flash('Anda telah logout!', 'info')
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    """Serve the main page"""
    return render_template('index.html', username=session.get('username'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """API endpoint for sentiment prediction"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'No text provided in request'
            }), 400
        
        text = data['text']
        
        # Validate input
        if not text or not text.strip():
            return jsonify({
                'success': False,
                'error': 'Text is empty'
            }), 400
        
        if len(text.strip()) < 5:
            return jsonify({
                'success': False,
                'error': 'Text is too short (minimum 5 characters)'
            }), 400
        
        # Make prediction
        svm_result, gnb_result, error = predict_sentiment(text)
        
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 500
        
        if svm_result is None or gnb_result is None:
            return jsonify({
                'success': False,
                'error': 'Prediction failed'
            }), 500
        
        # Return successful prediction
        return jsonify({
            'success': True,
            'predictions': {
                'svm': svm_result,
                'gnb': gnb_result
            },
            'input_text': text[:100] + ('...' if len(text) > 100 else '')  # Truncate for response
        })
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """Handle file upload and process CSV"""
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('Tidak ada file yang dipilih!', 'danger')
            return redirect(url_for('index'))
        
        file = request.files['file']
        
        # If user does not select file, browser also submits empty part without filename
        if file.filename == '':
            flash('Tidak ada file yang dipilih!', 'danger')
            return redirect(url_for('index'))
        
        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save file
            file.save(file_path)
            
            # Process the CSV file
            results = process_csv_file(file_path)
            
            # Clean up uploaded file
            try:
                os.remove(file_path)
            except:
                pass
            
            # Render results page
            return render_template('results.html', results=results, filename=filename)
        
        else:
            flash('Format file tidak didukung! Hanya file CSV yang diperbolehkan.', 'danger')
            return redirect(url_for('index'))
            
    except Exception as e:
        logger.error(f"Error in file upload: {str(e)}")
        flash(f'Terjadi kesalahan saat memproses file: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        # Check if models are loaded
        models_loaded = all([
            svm_model is not None,
            gnb_model is not None,
            vectorizer is not None,
            stemmer is not None,
            stop_remover is not None
        ])
        
        return jsonify({
            'status': 'healthy' if models_loaded else 'unhealthy',
            'models_loaded': models_loaded,
            'dependencies_ok': check_dependencies(),
            'python_version': sys.version
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

def print_startup_info():
    """Print startup information"""
    print("=" * 50)
    print("🚀 SENTIMENT ANALYSIS SERVER")
    print("=" * 50)
    print(f"📍 Server URL: http://localhost:5000")
    print(f"🔐 Login Page: http://localhost:5000/login")
    print(f"📊 API Endpoint: http://localhost:5000/predict")
    print(f"📁 File Upload: http://localhost:5000/upload")
    print(f"❤️  Health Check: http://localhost:5000/health")
    print("=" * 50)
    print("👤 Default Users:")
    print("  Username: admin | Password: password123")
    print("  Username: user  | Password: user123")
    print("=" * 50)
    print("📝 Usage:")
    print("  1. Login at http://localhost:5000/login")
    print("  2. Manual input: Enter Indonesian review text")
    print("  3. File upload: Upload CSV with review data")
    print("  4. View results and analysis")
    print("=" * 50)

if __name__ == '__main__':
    print("Starting Sentiment Analysis Server...")
    
    # Check if models directory exists
    if not os.path.exists('models'):
        print("❌ ERROR: 'models' directory not found!")
        print("\nPlease ensure the following files exist:")
        print("  - models/svm_model.pkl")
        print("  - models/gnb_model.pkl") 
        print("  - models/tfidf_vectorizer.pkl")
        print("\nCopy your model files to the 'models' folder and try again.")
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Load models on startup
    print("Loading machine learning models...")
    if not load_models():
        print("❌ Failed to load models. Please check the error messages above.")
        input("Press Enter to exit...")
        sys.exit(1)
    
    print("✅ All models loaded successfully!")
    print_startup_info()
    
    # Run the app
    try:
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        input("Press Enter to exit...")