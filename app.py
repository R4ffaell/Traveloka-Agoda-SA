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
from datetime import datetime
import numpy as np

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
        if not check_dependencies():
            return False
            
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        
        # ganti aja nanti jadi directory file nya
        model_dir = r"C:\Users\anakt\OneDrive\Desktop\Traveloka-Agoda-SA\models"


        model_files = [
            os.path.join(model_dir, "svm_model.pkl"),
            os.path.join(model_dir, "gnb_model.pkl"),
            os.path.join(model_dir, "tfidf_vectorizer.pkl")  
        ]

        
        # Check if all model files exist
        for file_path in model_files:
            if not os.path.exists(file_path):
                logger.error(f"Model file not found: {file_path}")
                print(f"ERROR: Model file not found - {file_path}")
                return False
        
        logger.info("Loading SVM model...")
        svm_model = joblib.load(os.path.join(model_dir, "svm_model.pkl"))
        
        logger.info("Loading GNB model...")
        gnb_model = joblib.load(os.path.join(model_dir, "gnb_model.pkl"))
        
        logger.info("Loading TF-IDF vectorizer...")
        vectorizer = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
        
        logger.info("Initializing stemmer...")
        stemmer = StemmerFactory().create_stemmer()
        
        logger.info("Initializing stopword remover...")
        stop_remover = StopWordRemoverFactory().create_stop_word_remover()
        
        logger.info("All models and preprocessing tools loaded successfully")
        print("✅ Models loaded:")
        print(f"   - SVM Model: {type(svm_model).__name__}")
        print(f"   - GNB Model: {type(gnb_model).__name__}")
        print(f"   - Vectorizer: {type(vectorizer).__name__}")
        
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
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r"@\w+|#\w+", "", text)
        
        # Remove punctuation and special characters
        text = re.sub(r"[^\w\s]", " ", text)
        
        # Remove numbers
        text = re.sub(r"\d+", "", text)
        
        # Remove extra whitespace
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
        
        # Final cleanup
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    except Exception as e:
        logger.error(f"Error in advanced cleaning: {str(e)}")
        return text

def predict_sentiment(text):
    """Predict sentiment using both SVM and GNB models"""
    try:
        if not text or not text.strip():
            return None, None, "Input text is empty"
        
        if any(model is None for model in [svm_model, gnb_model, vectorizer, stemmer, stop_remover]):
            return None, None, "Models not loaded properly"
        
        # Preprocessing pipeline
        logger.debug(f"Original text: {text[:100]}...")
        
        # Basic preprocessing
        clean_text = preprocess_text(text)
        if not clean_text:
            return None, None, "Text became empty after preprocessing"
            
        logger.debug(f"After basic preprocessing: {clean_text[:100]}...")
        
        # Advanced preprocessing
        final_text = advanced_clean(clean_text)
        if not final_text:
            return None, None, "Text became empty after advanced cleaning"
            
        logger.debug(f"After advanced preprocessing: {final_text[:100]}...")
        
        # Vectorization
        try:
            tfidf_input = vectorizer.transform([final_text])
            logger.debug(f"TF-IDF shape: {tfidf_input.shape}")
        except Exception as e:
            logger.error(f"Error in vectorization: {str(e)}")
            return None, None, f"Vectorization error: {str(e)}"
        
        # Predictions
        try:
            # SVM prediction
            svm_pred = svm_model.predict(tfidf_input)[0]
            svm_proba = None
            if hasattr(svm_model, 'predict_proba'):
                svm_proba = svm_model.predict_proba(tfidf_input)[0]
                
            # GNB prediction (requires dense array)
            gnb_input = tfidf_input.toarray() if hasattr(tfidf_input, 'toarray') else tfidf_input
            gnb_pred = gnb_model.predict(gnb_input)[0]
            gnb_proba = None
            if hasattr(gnb_model, 'predict_proba'):
                gnb_proba = gnb_model.predict_proba(gnb_input)[0]
            
            logger.debug(f"SVM prediction: {svm_pred}, GNB prediction: {gnb_pred}")
            
        except Exception as e:
            logger.error(f"Error in model prediction: {str(e)}")
            return None, None, f"Prediction error: {str(e)}"
        
        # Map predictions to labels
        # Adjust this mapping based on your model's output format
        if isinstance(svm_pred, (int, np.integer)):
            # If predictions are integers (0, 1, 2)
            label_map = {0: "Negatif", 1: "Netral", 2: "Positif"}
            svm_result = label_map.get(svm_pred, f"Unknown({svm_pred})")
            gnb_result = label_map.get(gnb_pred, f"Unknown({gnb_pred})")
        else:
            # If predictions are strings
            svm_result = str(svm_pred).title()
            gnb_result = str(gnb_pred).title()
        
        # Add confidence scores if available
        result_data = {
            'svm_label': svm_result,
            'gnb_label': gnb_result,
            'svm_confidence': float(max(svm_proba)) if svm_proba is not None else None,
            'gnb_confidence': float(max(gnb_proba)) if gnb_proba is not None else None,
            'preprocessed_text': final_text[:100] + ('...' if len(final_text) > 100 else '')
        }
        
        return svm_result, gnb_result, None, result_data
        
    except Exception as e:
        error_msg = f"Error in prediction: {str(e)}"
        logger.error(error_msg)
        return None, None, error_msg, None

def parse_date(date_string):
    """Parse date string with multiple format support"""
    if not isinstance(date_string, str):
        return None
    
    formats_to_try = [
        '%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y', '%m/%d/%Y',
        '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f',
        '%d/%m/%Y %H:%M', '%m/%d/%Y %H:%M'
    ]
    
    for fmt in formats_to_try:
        try:
            return datetime.strptime(date_string.strip(), fmt).isoformat()
        except (ValueError, TypeError):
            continue
    return None

def process_csv_file(file_path):
    """Process CSV file and predict sentiment using ML models"""
    results = []
    try:
        # Read CSV file
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        
        logger.info(f"CSV loaded with {len(df)} rows and columns: {list(df.columns)}")
        
        # Find text and date columns dynamically
        text_column = None
        date_column = None
        
        for col in df.columns:
            col_lower = col.lower().strip()
            if not text_column and col_lower in ['text', 'review', 'ulasan', 'komentar', 'comment', 'content', 'isi']:
                text_column = col
            if not date_column and col_lower in ['date', 'tanggal', 'timestamp', 'waktu', 'created_at', 'time']:
                date_column = col
        
        # Use first column as text if no specific text column found
        if not text_column:
            text_column = df.columns[0]
            logger.warning(f"No text column found, using first column: '{text_column}'")
        
        logger.info(f"Using text column: '{text_column}', date column: '{date_column}'")
        
        # Process each row
        total_rows = len(df)
        processed_count = 0
        
        for index, row in df.iterrows():
            try:
                # Get text content
                text = str(row[text_column]).strip() if pd.notna(row[text_column]) else ""
                
                # Parse date
                timestamp = None
                if date_column and date_column in row and pd.notna(row[date_column]):
                    timestamp = parse_date(str(row[date_column]))
                
                if not timestamp:
                    timestamp = datetime.now().isoformat()
                
                # Skip empty or invalid text
                if not text or text.lower() in ['nan', 'null', 'none', '']:
                    results.append({
                        'text': 'Empty text',
                        'svm': 'Skipped',
                        'gnb': 'Skipped',
                        'timestamp': timestamp,
                        'confidence_svm': None,
                        'confidence_gnb': None
                    })
                    continue
                
                # Predict sentiment
                svm_result, gnb_result, error, extra_data = predict_sentiment(text)
                
                if error:
                    logger.warning(f"Prediction error for row {index}: {error}")
                    results.append({
                        'text': text[:100],
                        'svm': 'Error',
                        'gnb': 'Error',
                        'timestamp': timestamp,
                        'error': error,
                        'confidence_svm': None,
                        'confidence_gnb': None
                    })
                else:
                    results.append({
                        'text': text[:100],
                        'svm': svm_result,
                        'gnb': gnb_result,
                        'timestamp': timestamp,
                        'confidence_svm': extra_data.get('svm_confidence') if extra_data else None,
                        'confidence_gnb': extra_data.get('gnb_confidence') if extra_data else None,
                        'preprocessed': extra_data.get('preprocessed_text') if extra_data else None
                    })
                
                processed_count += 1
                
                # Log progress every 100 rows
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count}/{total_rows} rows")
                    
            except Exception as e:
                logger.error(f"Error processing row {index}: {str(e)}")
                results.append({
                    'text': str(row.get(text_column, ''))[:100],
                    'svm': 'Error',
                    'gnb': 'Error',
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    'confidence_svm': None,
                    'confidence_gnb': None
                })

        logger.info(f"Finished processing {processed_count} rows")
        
        # Calculate statistics
        successful_predictions = [r for r in results if r['svm'] not in ['Error', 'Skipped']]
        stats = {
            'total_rows': total_rows,
            'processed_rows': processed_count,
            'successful_predictions': len(successful_predictions),
            'errors': len([r for r in results if r['svm'] == 'Error']),
            'skipped': len([r for r in results if r['svm'] == 'Skipped'])
        }
        
        return {
            'success': True, 
            'results': results,
            'stats': stats,
            'message': f"Successfully processed {len(successful_predictions)} out of {total_rows} rows"
        }

    except Exception as e:
        error_msg = f"Error processing CSV: {str(e)}"
        logger.error(error_msg)
        return {'success': False, 'error': error_msg}

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

@app.route('/dashboard')
@login_required
def dashboard():
    """Serve the dashboard page"""
    username = session.get('username', 'User') 
    return render_template('dashboard.html', username=username)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """API endpoint for sentiment prediction"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'success': False, 'error': 'No text provided in request'}), 400
        
        text = data['text']
        if not text or not text.strip():
            return jsonify({'success': False, 'error': 'Text is empty'}), 400
        if len(text.strip()) < 5:
            return jsonify({'success': False, 'error': 'Text is too short (minimum 5 characters)'}), 400
        
        svm_result, gnb_result, error, extra_data = predict_sentiment(text)
        
        if error:
            return jsonify({'success': False, 'error': error}), 500
        if svm_result is None or gnb_result is None:
            return jsonify({'success': False, 'error': 'Prediction failed'}), 500
        
        response_data = {
            'success': True,
            'predictions': {
                'svm': svm_result, 
                'gnb': gnb_result
            },
            'input_text': text[:100] + ('...' if len(text) > 100 else ''),
            'preprocessed_text': extra_data.get('preprocessed_text') if extra_data else None,
            'confidence': {
                'svm': extra_data.get('svm_confidence') if extra_data else None,
                'gnb': extra_data.get('gnb_confidence') if extra_data else None
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

# Global variable for last upload results
last_upload_results = []

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    global last_upload_results
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Tidak ada file yang dipilih!'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Tidak ada file yang dipilih!'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save file
            file.save(file_path)
            logger.info(f"File saved to: {file_path}")
            
            # Process file
            response_data = process_csv_file(file_path)
            
            # Store results globally
            if response_data.get('success'):
                last_upload_results = response_data['results']
                logger.info(f"Stored {len(last_upload_results)} results globally")
            else:
                last_upload_results = []
            
            # Clean up file
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up file: {file_path}")
            except Exception as e:
                logger.warning(f"Could not remove file {file_path}: {e}")
            
            return jsonify(response_data)
        
        else:
            return jsonify({'success': False, 'error': 'Format file tidak didukung! Hanya file CSV.'}), 400
            
    except Exception as e:
        logger.error(f"Error in file upload: {str(e)}")
        return jsonify({'success': False, 'error': f'Terjadi kesalahan internal: {str(e)}'}), 500

@app.route('/api/sentiment')
@login_required
def api_sentiment():
    """API endpoint to get sentiment data"""
    global last_upload_results
    
    try:
        if not last_upload_results:
            return jsonify({
                'success': False, 
                'data': [], 
                'message': 'No data available. Please upload a CSV file first.'
            }), 404
        
        # Calculate summary statistics
        successful_data = [item for item in last_upload_results if item['svm'] not in ['Error', 'Skipped']]
        
        stats = {
            'total': len(last_upload_results),
            'successful': len(successful_data),
            'positive': len([item for item in successful_data if 'positif' in item['svm'].lower()]),
            'negative': len([item for item in successful_data if 'negatif' in item['svm'].lower()]),
            'neutral': len([item for item in successful_data if 'netral' in item['svm'].lower()]),
            'errors': len([item for item in last_upload_results if item['svm'] == 'Error']),
            'skipped': len([item for item in last_upload_results if item['svm'] == 'Skipped'])
        }
        
        return jsonify({
            'success': True, 
            'data': last_upload_results,
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in api_sentiment: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        models_loaded = all([
            svm_model is not None, gnb_model is not None,
            vectorizer is not None, stemmer is not None,
            stop_remover is not None
        ])
        
        model_info = {}
        if models_loaded:
            model_info = {
                'svm_model_type': type(svm_model).__name__,
                'gnb_model_type': type(gnb_model).__name__,
                'vectorizer_type': type(vectorizer).__name__,
                'total_uploaded_records': len(last_upload_results) if last_upload_results else 0
            }
        
        return jsonify({
            'status': 'healthy' if models_loaded else 'unhealthy',
            'models_loaded': models_loaded,
            'dependencies_ok': check_dependencies(),
            'python_version': sys.version,
            'model_info': model_info
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

def print_startup_info():
    """Print startup information"""
    print("=" * 60)
    print("🚀 SENTIMENT ANALYSIS SERVER - ML POWERED")
    print("=" * 60)
    print(f"📍 Server URL: http://localhost:5000")
    print(f"🔐 Login Page: http://localhost:5000/login")
    print(f"📊 Dashboard: http://localhost:5000/dashboard")
    print(f"📊 API Predict: http://localhost:5000/predict")
    print(f"📁 File Upload: http://localhost:5000/upload")
    print(f"❤️  Health Check: http://localhost:5000/health")
    print("=" * 60)
    print("👤 Default Users:")
    print("  Username: admin | Password: password123")
    print("  Username: user  | Password: user123")
    print("=" * 60)
    print("🤖 ML Models:")
    print(f"  SVM Model: {type(svm_model).__name__ if svm_model else 'Not loaded'}")
    print(f"  GNB Model: {type(gnb_model).__name__ if gnb_model else 'Not loaded'}")
    print(f"  Vectorizer: {type(vectorizer).__name__ if vectorizer else 'Not loaded'}")
    print("=" * 60)
    print("📝 Usage:")
    print("  1. Login at http://localhost:5000/login")
    print("  2. Go to dashboard: http://localhost:5000/dashboard")
    print("  3. Upload CSV file with text/review data")
    print("  4. View ML-powered sentiment analysis results")
    print("=" * 60)

if __name__ == '__main__':
    print("Starting ML-Powered Sentiment Analysis Server...")
    
    # Check models directory
    if not os.path.exists('models'):
        print("❌ ERROR: 'models' directory not found!")
        print("\nPlease ensure the following files exist:")
        print("  - models/svm_model.joblib")
        print("  - models/gnb_model.joblib") 
        print("  - models/tfidf.joblib")
        print("\nCopy your trained model files to the 'models' folder and try again.")
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Load models
    print("Loading machine learning models...")
    if not load_models():
        print("❌ Failed to load models. Please check the error messages above.")
        print("\nMake sure you have:")
        print("1. Trained SVM and GNB models saved as .joblib files")
        print("2. TF-IDF vectorizer saved as .joblib file")
        print("3. Sastrawi library installed (pip install Sastrawi)")
        input("Press Enter to exit...")
        sys.exit(1)
    
    print("✅ All models loaded successfully!")
    print_startup_info()
    
    try:
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # Set to False in production
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        input("Press Enter to exit...")