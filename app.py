# libbary
import numpy as np  # linear algebra
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from joblib import load
import csv
from sklearn.metrics import accuracy_score


"""# import pickle
import pickle"""

'model = pickle.load(open("model.pkl", "rb"))'
app = Flask(__name__)

from joblib import dump

# Load the trained model
model = load("model.joblib")

# Load the trained model
model = load("model2.joblib")

# Feature names
feature_names = [
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
    "A7",
    "A8",
    "A9",
    "A10",
    "Jaundice",
    "Family_mem_with_ASD",
]

"""# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)"""


# Home Page
@app.route("/")
def index():
    print(type(model))
    return render_template("index.html")


# Home Page
@app.route("/screening")
def screening():
    return render_template("screening.html")


@app.route("/result", methods=["POST"])
def predict():
    # mengambil jawaban kuesioner dari form
    jawaban = [
        int(request.form["A1"]),
        int(request.form["A2"]),
        int(request.form["A3"]),
        int(request.form["A4"]),
        int(request.form["A5"]),
        int(request.form["A6"]),
        int(request.form["A7"]),
        int(request.form["A8"]),
        int(request.form["A9"]),
        int(request.form["A10"]),
        int(request.form["Jaundice"]),
        int(request.form["Family_mem_with_ASD"]),
    ]

    # mengubah jawaban menjadi array numpy
    jawaban = np.array(jawaban).reshape(1, -1)

    # membuat prediksi dengan model random forest
    y_pred = model.predict(jawaban)
    "prediksi = model.predict(jawaban)"

    # Save the user input to a dictionary
    user_input = {feature_names[i]: jawaban[0, i] for i in range(len(feature_names))}

    # Save the user input to a CSV file
    with open("user_input.csv", mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=feature_names + ["Prediction"])
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow({**user_input, "Prediction": y_pred[0]})

    """ # Calculate the percentage of each class in the prediction
    class_labels = model.classes_
    pred_probabilities = model.predict_proba(jawaban)[
        0
    ]  # Dapatkan probabilitas kelas dari array
    class_percentages = {
        class_labels[i]: round(pred_probabilities[i] * 100, 2)
        for i in range(len(class_labels))
    }"""

    # Calculate the percentage of each class in the prediction
    class_labels = model.classes_
    pred_probabilities = model.predict_proba(jawaban)[
        0
    ]  # Dapatkan probabilitas kelas dari array
    class_percentages = {
        class_labels[i]: round(pred_probabilities[i] * 100, 2)
        for i in range(len(class_labels))
    }

    # Find the class with the highest probability and its corresponding percentage
    highest_class = max(class_percentages, key=class_percentages.get)
    highest_percentage = class_percentages[highest_class]

    # Mencetak nilai class_percentages ke konsol
    print(class_percentages)
    """# Make a test prediction
    test_data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1]])  # Example input data
    y_pred = model.predict(test_data)
    print(y_pred)"""
    # Simpan hasil prediksi dan persentase kelas tertinggi ke dalam satu variabel
    result = {
        "prediksi": y_pred[0],
        "user_input": user_input,
        "highest_class": highest_class,
        "highest_percentage": highest_percentage,
    }

    return render_template(
        "result.html",
        prediksi=y_pred[0],
        user_input=user_input,
        class_percentages=class_percentages,
        result=result,
    )


if __name__ == "__main__":
    app.run(debug=True)
