# Duplicate Question Pair Detection

A machine learning application that detects duplicate question pairs using advanced NLP features and scikit-learn models, deployed on Vercel.

## 🚀 Features

- **Fast Detection**: Uses pre-trained ML model for instant duplicate detection
- **Feature-Rich**: Analyzes token similarity, fuzzy matching, length features, and more
- **Beautiful UI**: Modern, responsive web interface
- **REST API**: Easy integration with other applications
- **Live Demo**: Deployed on Vercel - no setup required!

## 📋 Project Structure

```
.
├── api/                      # Vercel API endpoint
│   └── index.py             # Flask app (main entry point)
├── WEB/                      # Helper modules & models
│   ├── helper.py            # Feature extraction functions
│   ├── model.pkl            # Pre-trained model
│   ├── cv.pkl               # Count vectorizer
│   └── stopwords.pkl        # Stopwords list
├── public/                   # Frontend
│   └── index.html           # Interactive web UI
├── vercel.json              # Vercel configuration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🛠️ Installation & Setup

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo>
   cd Duplicate-Question-Pair
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run locally**
   ```bash
   python api/index.py
   ```
   
   Visit: `http://localhost:5000`

## 🌐 Deployment to Vercel

### Prerequisites
- GitHub account with your project repo
- Vercel account (free tier available)

### Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Vercel deployment setup"
   git push origin main
   ```

2. **Connect to Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Select your GitHub repository
   - Vercel auto-detects the Python project
   - Click "Deploy"

3. **Your app is live!**
   - Access at: `https://your-project.vercel.app`

## 📝 API Usage

### Check for Duplicate Questions

**Endpoint:** `POST /api/check`

**Request:**
```json
{
  "q1": "How do I read a file in Python?",
  "q2": "How can I read a file using Python?"
}
```

**Response:**
```json
{
  "q1": "How do I read a file in Python?",
  "q2": "How can I read a file using Python?",
  "is_duplicate": true,
  "probability_not_duplicate": 0.23,
  "probability_duplicate": 0.77
}
```

### Health Check

**Endpoint:** `GET /api/health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## 🔧 Technologies Used

- **Backend**: Flask, scikit-learn
- **NLP**: FuzzyWuzzy, BeautifulSoup, NLTK
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Deployment**: Vercel
- **Data Science**: Jupyter Notebooks (EDA & model training)

## 📊 Model Features

The model analyzes multiple feature categories:

- **Length Features** (3): Token count difference, average length, longest common substring
- **Token Features** (8): Common words, stopwords, token overlap
- **Fuzzy Features** (4): QRatio, partial ratio, token sort, token set
- **Basic Features** (7): Character length, word count, common words
- **Bag of Words**: TF-IDF vectors for both questions

## 📚 Notebooks

- `EDA.ipynb`: Exploratory data analysis on the Quora dataset
- `bow (basic-features).ipynb`: Feature engineering with bag-of-words
- `BOW.ipynb`: Model training and evaluation

## Project Structure
- `EDA.ipynb`: Performs exploratory data analysis (EDA) on the Quora dataset. It includes data loading, visualization, and preliminary analysis to understand the data's characteristics.
- `bow (basic-features).ipynb`: Prepares the dataset for modeling by applying feature engineering techniques like bag-of-words (BoW) to represent questions.
- `BOW.ipynb`: Implements machine learning models using the features prepared in the previous notebook. It focuses on setting up and evaluating different models to identify duplicate questions.

## Setup
To run this project, follow these steps:
1. **Clone the Repository**:
   ```
   git clone <repository-url>
   ```
2. **Install Required Libraries**:
   Ensure you have Python installed and proceed to install the required packages:
   ```
   pip install numpy pandas matplotlib seaborn sklearn
   ```
3. **Run the Notebooks**:
   Open the Jupyter Notebooks in an environment that supports IPython (like Jupyter Lab or VSCode) and execute them in order.

## Data
The dataset used in this project is the Quora question pairs dataset, which includes several thousands of question pairs along with a label indicating if they are duplicates.

## Contributing
Contributions to this project are welcome! Please feel free to fork the repository, make improvements, and submit a pull request.

## License
This project is open-sourced under the MIT license.

