Interactive Fake News Detector

An interactive web app that uses a machine learning model to classify news articles as "Real" or "Fake".

🚀 Live Demo Link

(This link will be added after deployment!)

Features

Real-Time Classification: Paste in any news text and get an instant prediction.

High Accuracy: Built on a LogisticRegression model trained on over 44,000 articles.

Clean Interface: Simple, fast, and easy-to-use UI built with Streamlit.

Robust Text Processing: Uses TfidfVectorizer to analyze word importance.

🤖 Tech Stack

Python: Core programming language.

Scikit-learn: For machine learning (model training and vectorization).

Pandas: For data loading and manipulation.

Streamlit: To build and deploy the interactive web app.

Git LFS: To manage and version-control the large model files.

⚙️ How to Run This Project Locally

Clone the repository:

git clone [https://github.com/your-username/fake-news-detector.git](https://github.com/your-username/fake-news-detector.git)
cd fake-news-detector


Set up Git LFS (This is crucial for downloading the models):

git lfs install
git lfs pull


Create and activate a virtual environment:

# For Mac/Linux
python3 -m venv venv
source vVenv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate


Install the required libraries:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py


The app will open in your browser at http://localhost:8501.

📂 Dataset

This model was trained on a combined dataset of two popular sources:

True.csv: ~21,000 real news articles from Reuters.

Fake.csv: ~23,000 fake news articles from various sources.