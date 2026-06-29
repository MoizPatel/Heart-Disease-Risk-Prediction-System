# Heart-Disease-Risk-Prediction-System

## 📝 Description
A machine learning-based clinical decision support system designed to predict cardiovascular disease risk. The application processes patient clinical data to provide an accurate, transparent, and actionable risk diagnosis.

## 🚀 Key Features
* **Machine Learning Engine**: Utilizes a Random Forest Classifier trained on 918 patient records to detect non-linear clinical patterns.
* **Explainable AI**: Implements a probabilistic "Confidence Score" using `predict_proba()`, moving beyond simple binary outputs to provide clinical transparency[cite: 3].
* **Interactive UI**: Built with Streamlit to allow medical professionals to input vitals and receive real-time risk reports[cite: 3].

## 🛠 Technologies Used
* **Python**: Core logic and data manipulation[cite: 3].
* **Scikit-Learn**: Implementation of the Random Forest ensemble architecture[cite: 3].
* **Pandas**: Efficient handling of clinical datasets[cite: 3].
* **Streamlit**: Frontend framework for web deployment[cite: 3].

## ⚙️ How to Run
1. Clone this repository: `git clone https://github.com/MoizPatel/heart-disease-prediction-system`
2. Install the required libraries: `pip install -r requirements.txt`
3. Launch the application: `streamlit run src/app.py`
