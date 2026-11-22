document.addEventListener('DOMContentLoaded', () => {
    const symptoms = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
        'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
        'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
        'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
        'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
        'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
        'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
        'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
        'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
        'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
        'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
        'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
        'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
        'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
        'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
        'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
        'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf',
        'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
        'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
        'yellow_crust_ooze']; // Continue with all symptoms
    symptoms.forEach(symptom => {
        document.querySelectorAll('select').forEach(select => {
            let option = document.createElement('option');
            option.value = symptom;
            option.text = symptom
                .replace(/_/g, ' ')
                .replace(/\s+/g, ' ')
                .trim()
                .replace(/^./, character => character.toUpperCase());
            select.add(option);
        });
    });
});

// script.js
const predictions = {};
let blinkInterval;
function showToast(message, color) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.style.backgroundColor = color;
    toast.className = 'show';

    setTimeout(() => {
        toast.className = toast.className.replace('show', '');
    }, 3000);  // Toast duration in milliseconds
}

function predict(modelType) {
    const symptoms = [
        document.getElementById('symptom1').value,
        document.getElementById('symptom2').value,
        document.getElementById('symptom3').value,
        document.getElementById('symptom4').value,
        document.getElementById('symptom5').value,
    ];

    const selectedSymptoms = symptoms.filter(symptom => symptom !== "");
    if (selectedSymptoms.length < 3) {
        showToast('Please select at least three symptom.', 'yellowgreen');
        return;
    }

    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model_type: modelType,
            symptoms: symptoms
        }),
    })
        .then(response => {
            if (response.status === 200) {
                return response.json();
            } else if (response.status === 404) {
                throw new Error('404 Not Found');
            } else if (response.status === 500) {
                throw new Error('500 Internal Server Error');
            } else {
                throw new Error('Unexpected status code: ' + response.status);
            }
        })
        .then(data => {
            let resultDivId = '';
            predictions[modelType] = data.prediction;
            if (modelType === 'DecisionTree') {
                resultDivId = 'resultDecisionTree';
            } else if (modelType === 'RandomForest') {
                resultDivId = 'resultRandomForest';
            } else if (modelType === 'NaiveBayes') {
                resultDivId = 'resultNaiveBayes';
            }

            const resultElement = document.getElementById(resultDivId);
            resultElement.textContent = 'Prediction: ' + data.prediction;
            resultElement.style.display = 'block';
            showToast('Success: Prediction received! ' ,'green');
            if (Object.keys(predictions).length === 3) { displayFinalPrediction(); }
        })
        .catch((error) => {
            console.error('Error:', error);
            if (error.message.includes('404')) {
                showToast('Error: 404 Not Found', 'yellow');
            } else if (error.message.includes('500')) {
                showToast('Error: 500 Internal Server Error', 'red');
            } else {
                showToast('Error: ' + error.message, 'orange');
            }
        });
}

function clearFields() {
    document.getElementById('name').value = '';
    document.querySelectorAll('select').forEach(select => {
        select.value = '';
    });
    Object.keys(predictions).forEach(modelType => delete predictions[modelType]);
    ['resultDecisionTree', 'resultRandomForest', 'resultNaiveBayes'].forEach(id => {
        const result = document.getElementById(id);
        result.textContent = '';
        result.style.display = 'none';
    });
    const finalPrediction = document.getElementById('finalPrediction');
    finalPrediction.textContent = '';
    finalPrediction.style.display = 'none';
}

function displayFinalPrediction() {
    const predictionCounts = {};
    // Count the occurrences of each prediction
    Object.values(predictions).forEach(prediction => { if (predictionCounts[prediction]) { predictionCounts[prediction]++; } else { predictionCounts[prediction] = 1; } });
    // Determine the most frequent prediction 
    let finalPrediction = null;
    let maxCount = 0;
    for (const [prediction, count] of Object.entries(predictionCounts)) {
        if (count > maxCount) {
            finalPrediction = prediction; maxCount = count;
        }
    }
    const userName = document.getElementById('name').value;
    const finalPredictionElement = document.getElementById('finalPrediction');

    finalPredictionElement.textContent =
        `${userName ? `${userName}, ` : ''}our models suggest ${finalPrediction}. This is not a diagnosis—please consult a healthcare professional.`;
    finalPredictionElement.style.display = 'block';
    finalPredictionElement.classList.add('is-highlighted');
    requestAnimationFrame(() => {
        finalPredictionElement.scrollIntoView({
            behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
            block: 'center',
        });
    });
}
