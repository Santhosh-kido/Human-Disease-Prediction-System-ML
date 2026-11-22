# Human Disease Prediction System

A web application that predicts a possible disease from the symptoms a user selects. The browser sends the symptoms to a Flask API, three scikit-learn models each make a prediction, and the page shows every model's result along with a final majority-vote prediction.

> **Disclaimer:** This is a learning project. It is not a medical tool and must not be used for diagnosis. Always consult a doctor.

## Features

- Patient name field and five symptom dropdowns, filled from a list of 132 symptoms
- At least 3 symptoms required before a prediction is made
- Three models: Decision Tree, Random Forest and Gaussian Naive Bayes
- Each model's prediction shown separately
- Final prediction chosen by majority vote of the three models
- Clear button to reset all fields and results
- Toast notifications for success and errors
- Flask REST API with CORS enabled and a health-check endpoint

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask, Flask-CORS |
| Machine learning | scikit-learn, pandas, NumPy |
| Models | Decision Tree, Random Forest, Gaussian Naive Bayes |
| Data | CSV files (`Training.csv`, `Testing.csv`) |

## How It Works

```
Browser (index.html, script.js)
  -> user picks a name and 3 to 5 symptoms
  -> POST /predict  { "model_type": "DecisionTree", "symptoms": [...] }
  -> Flask turns the symptoms into a 132-value vector (1 = present, 0 = absent)
  -> the chosen model predicts a disease label (0 to 40)
  -> the label is mapped back to the disease name
  -> JSON response  { "prediction": "Heart attack" }
  -> after all three models respond, the page shows the majority vote
```

## Dataset

- `Training.csv`: 4,920 rows, 132 symptom columns and a `prognosis` column
- `Testing.csv`: 41 rows, one example for each disease
- 41 diseases in total. Each symptom is `1` if present and `0` if absent.
- Source: <state honestly, for example "public Kaggle dataset: <name and link>">

## Project Structure

```
Human-Disease-Prediction/
├── frontend/
│   ├── index.html       # Page layout and result areas
│   ├── script.js        # Symptom dropdowns, API calls, majority vote
│   └── styles.css       # Styling
├── backend/
│   ├── app.py           # Flask API and model code
│   ├── Training.csv
│   ├── Testing.csv
│   └── requirements.txt
└── README.md
```

## Run Locally

**Prerequisites:** Python 3.9 or newer, pip

**1. Clone and install**
```
git clone git@github.com:Santhosh-kido/Human-Disease-Prediction.git
cd Human-Disease-Prediction
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
```

**2. Start the backend**
```
cd backend
python app.py
```
The API runs on http://127.0.0.1:5000.

**3. Start the frontend** (in a second terminal)
```
cd frontend
python -m http.server 5500
```
Open http://localhost:5500.

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health-check` | Confirms the server is running |
| POST | `/predict` | Returns a predicted disease |

**Request body for `/predict`**
```json
{
  "model_type": "RandomForest",
  "symptoms": ["chest_pain", "fatigue", "dizziness"]
}
```
`model_type` must be `DecisionTree`, `RandomForest` or `NaiveBayes`. An invalid value returns a JSON error.

**Response**
```json
{ "prediction": "Heart attack" }
```

## Model Results

| Model | Accuracy |
|---|---|
| Decision Tree | <value> |
| Random Forest | <value> |
| Gaussian Naive Bayes | <value> |

Accuracy is measured on <describe the evaluation: for example a train/test split of `Training.csv`>.
## Screenshots
<img width="1912" height="942" alt="image" src="https://github.com/user-attachments/assets/0cbd3bb7-1979-4a8d-b09f-17ed886d2fc2" />
![Uploading image.png…]()


## Known Limitations

- The models are trained again on every `/predict` request. This is fine for a demo, but a real service should train once, save the model and load it at startup.
- The dataset is simple and symptom-based, so the predictions are not clinically reliable.
- Random Forest results can differ between runs unless a random seed is set.
- The Flask development server is used, so it is not meant for public deployment.

## Future Improvements

- Train once and save the models with `joblib`
- Use `Training.csv` for training and a separate split for evaluation, and report precision and recall
- Add input validation and a clearer medical warning in the UI
- Deploy the API and frontend online

## Author

**Santhosh R**
B.E. Computer Science and Engineering
[LinkedIn](https://www.linkedin.com/in/santhosh-r10/) | [GitHub](https://github.com/Santhosh-kido)
