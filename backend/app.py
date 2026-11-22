from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from pathlib import Path

app = Flask(__name__)
CORS(app)
data_dir = Path(__file__).resolve().parent

# Your existing data preparation code here...
pd.set_option('future.no_silent_downcasting', True)
# Load data
l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']
disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

# Keep one entry per model feature; CSV column names must be unique.
l1 = list(dict.fromkeys(l1))

df = pd.read_csv(data_dir / "Training.csv")
#df.replace({'prognosis': {d: i for i, d in enumerate(disease)}}, inplace=True)
df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)
df = df.infer_objects(copy=False)
X = df[l1]
y = df[["prognosis"]]
np.ravel(y)

tr = pd.read_csv(data_dir / "Testing.csv")
#tr.replace({'prognosis': {d: i for i, d in enumerate(disease)}}, inplace=True)
tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)
tr = tr.infer_objects(copy=False)
X_test = tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)

@app.route('/health-check', methods=['GET'])
def health_check():
    return 'Backend code is up and running!'


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model_type = data.get('model_type')
    symptoms = data.get('symptoms')
    
    l2 = [1 if s in symptoms else 0 for s in l1]
    if model_type == 'DecisionTree':
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X, np.ravel(y))

        y_pred=clf.predict(X_test)
        acc = ((accuracy_score(y_test, y_pred))*100)
        print("decision tree:",acc)
        # print(accuracy_score(y_test, y_pred,normalize=False))
    elif model_type == 'RandomForest':
        clf = RandomForestClassifier()
        clf = clf.fit(X, np.ravel(y))

        y_pred=clf.predict(X_test)
        acc = ((accuracy_score(y_test, y_pred))*100)
        print("Random forest:",acc)
        # print(accuracy_score(y_test, y_pred,normalize=False))
    elif model_type == 'NaiveBayes':
        clf = GaussianNB()
        clf = clf.fit(X, np.ravel(y))

        y_pred=clf.predict(X_test)
        acc = ((accuracy_score(y_test, y_pred))*100)
        print("Naive bayes:",acc)
        # print(accuracy_score(y_test, y_pred,normalize=False))
    else:
        return jsonify({'error': 'Invalid model type'})
        clf.fit(X, np.ravel(y))
    prediction = clf.predict([l2])
    disease_predicted = disease[prediction[0]] if prediction[0] in range(len(disease)) else "Not Found"

    return jsonify({'prediction': disease_predicted})

if __name__ == '__main__':
    app.run(debug=True)


# # Fit and predict using Decision Tree
# clf_dt = tree.DecisionTreeClassifier()
# clf_dt = clf_dt.fit(X, y)
# y_pred_dt = clf_dt.predict(X_test)
# accuracy_dt = accuracy_score(y_test, y_pred_dt)
# print(f'Accuracy of Decision Tree: {accuracy_dt * 100:.2f}%')

# # Fit and predict using Random Forest
# clf_rf = RandomForestClassifier()
# clf_rf = clf_rf.fit(X, y)
# y_pred_rf = clf_rf.predict(X_test)
# accuracy_rf = accuracy_score(y_test, y_pred_rf)
# print(f'Accuracy of Random Forest: {accuracy_rf * 100:.2f}%')

# # Fit and predict using Naive Bayes
# clf_nb = GaussianNB()
# clf_nb = clf_nb.fit(X, y)
# y_pred_nb = clf_nb.predict(X_test)
# accuracy_nb = accuracy_score(y_test, y_pred_nb)
# print(f'Accuracy of Naive Bayes: {accuracy_nb * 100:.2f}%')
