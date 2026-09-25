from pathlib import Path
import os
import pickle
import sqlite3
from threading import Timer
import webbrowser

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from werkzeug.security import check_password_hash, generate_password_hash
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgboost_app_model.sav"
DATASET_PATH = BASE_DIR / "dataset.csv"
DB_PATH = BASE_DIR / "heart_disease.db"
FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalch",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
FORM_FEATURE_KEYS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
NUMERIC_FEATURES = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_FEATURES]

app = Flask(__name__, template_folder=".")
app.secret_key = os.environ.get("SECRET_KEY", "heart-disease-dev-secret")


def open_browser() -> None:
    webbrowser.open_new("http://127.0.0.1:5000/")


def get_db_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with get_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                patient_name TEXT,
                patient_email TEXT,
                age REAL NOT NULL,
                sex TEXT NOT NULL,
                cp TEXT NOT NULL,
                trestbps REAL NOT NULL,
                chol REAL NOT NULL,
                fbs TEXT NOT NULL,
                restecg TEXT NOT NULL,
                thalach REAL NOT NULL,
                exang TEXT NOT NULL,
                oldpeak REAL NOT NULL,
                slope TEXT NOT NULL,
                ca REAL NOT NULL,
                thal TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                probability REAL NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        )
        existing_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(predictions)").fetchall()
        }
        if "patient_name" not in existing_columns:
            connection.execute("ALTER TABLE predictions ADD COLUMN patient_name TEXT")
        if "patient_email" not in existing_columns:
            connection.execute("ALTER TABLE predictions ADD COLUMN patient_email TEXT")


def get_user_by_email(email: str) -> sqlite3.Row | None:
    with get_db_connection() as connection:
        return connection.execute(
            "SELECT * FROM users WHERE lower(email) = lower(?)", (email,)
        ).fetchone()


def create_user(full_name: str, email: str, password: str) -> None:
    with get_db_connection() as connection:
        connection.execute(
            """
            INSERT INTO users (full_name, email, password_hash)
            VALUES (?, ?, ?)
            """,
            (full_name.strip(), email.strip().lower(), generate_password_hash(password)),
        )


def save_prediction(form_data: dict, prediction: int, probability: float) -> None:
    values = {key: str(form_data.get(key, "")) for key in FORM_FEATURE_KEYS}
    user_name = session.get("user_name")
    user_email = session.get("user_email")
    with get_db_connection() as connection:
        if session.get("user_id") and (not user_name or not user_email):
            user = connection.execute(
                "SELECT full_name, email FROM users WHERE id = ?", (session["user_id"],)
            ).fetchone()
            if user:
                user_name = user["full_name"]
                user_email = user["email"]

        connection.execute(
            """
            INSERT INTO predictions (
                user_id, patient_name, patient_email, age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                exang, oldpeak, slope, ca, thal, prediction, probability
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.get("user_id"),
                user_name or "API Patient",
                user_email or "Not provided",
                float(values["age"]),
                "Male" if values["sex"] == "1" else "Female",
                values["cp"],
                float(values["trestbps"]),
                float(values["chol"]),
                values["fbs"],
                values["restecg"],
                float(values["thalach"]),
                values["exang"],
                float(values["oldpeak"]),
                values["slope"],
                float(values["ca"]),
                values["thal"],
                prediction,
                round(probability, 2),
            ),
        )


def login_required():
    if not session.get("user_id"):
        return redirect(url_for("home"))
    return None


def build_training_frame() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATASET_PATH)
    df["thal"] = df["thal"].replace({"reversable defect": "reversible defect"})
    df["target"] = (df["num"] > 0).astype(int)
    x = df[FEATURE_COLUMNS].copy()
    y = df["target"]
    return x, y


def build_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    objective="binary:logistic",
                    n_estimators=600,
                    max_depth=4,
                    learning_rate=0.03,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=2,
                    gamma=0.1,
                    reg_lambda=2.0,
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=42,
                ),
            ),
        ]
    )


def load_or_train_model() -> Pipeline:
    if MODEL_PATH.exists():
        with MODEL_PATH.open("rb") as model_file:
            return pickle.load(model_file)

    x, y = build_training_frame()
    x_train, _, y_train, _ = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    trained_model = build_model()
    trained_model.fit(x_train, y_train)

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(trained_model, model_file)

    return trained_model


def build_input_frame(data: dict) -> pd.DataFrame:
    def get_value(key: str) -> str:
        value = data.get(key)
        if value in (None, ""):
            raise ValueError(f"Missing required field: {key}")
        return str(value)

    return pd.DataFrame(
        [
            {
                "age": float(get_value("age")),
                "sex": "Male" if get_value("sex") == "1" else "Female",
                "cp": {
                    "0": "typical angina",
                    "1": "atypical angina",
                    "2": "non-anginal",
                    "3": "asymptomatic",
                }[get_value("cp")],
                "trestbps": float(get_value("trestbps")),
                "chol": float(get_value("chol")),
                "fbs": bool(int(float(get_value("fbs")))),
                "restecg": {
                    "0": "normal",
                    "1": "st-t abnormality",
                    "2": "lv hypertrophy",
                }[get_value("restecg")],
                "thalch": float(get_value("thalach")),
                "exang": bool(int(float(get_value("exang")))),
                "oldpeak": float(get_value("oldpeak")),
                "slope": {
                    "0": "upsloping",
                    "1": "flat",
                    "2": "downsloping",
                }[get_value("slope")],
                "ca": float(get_value("ca")),
                "thal": {
                    "1": "normal",
                    "2": "fixed defect",
                    "3": "reversible defect",
                }[get_value("thal")],
            }
        ]
    )


def normalize_api_payload(data: dict | None) -> dict:
    if not data:
        raise ValueError("Request body is empty.")

    if isinstance(data.get("features"), dict):
        return data["features"]

    if isinstance(data.get("features"), list):
        values = data["features"]
        if len(values) != len(FORM_FEATURE_KEYS):
            raise ValueError("features list must contain exactly 13 values.")
        return dict(zip(FORM_FEATURE_KEYS, values))

    return data


init_db()
model = load_or_train_model()


@app.route("/style.css")
def serve_style():
    return send_from_directory(BASE_DIR, "style.css")


@app.route("/", methods=["GET", "POST"], endpoint="home")
def home():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        user = get_user_by_email(email)

        if user is None or not check_password_hash(user["password_hash"], password):
            return render_template(
                "login.html",
                error_message="Invalid email or password.",
            )

        session["user_id"] = user["id"]
        session["user_name"] = user["full_name"]
        session["user_email"] = user["email"]
        return redirect(url_for("predict"))
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if password != confirm_password:
            return render_template(
                "signup.html",
                error_message="Passwords do not match.",
            )

        if get_user_by_email(email):
            return render_template(
                "signup.html",
                error_message="An account already exists for this email.",
            )

        create_user(full_name, email, password)
        return render_template(
            "login.html",
            signup_message="Registration complete. Please log in to continue.",
        )
    return render_template("signup.html")


@app.route("/predict", methods=["GET", "POST"], endpoint="predict")
def predict():
    login_redirect = login_required()
    if login_redirect:
        return login_redirect

    if request.method == "GET":
        return render_template("predict.html")

    try:
        input_frame = build_input_frame(request.form)
        prediction = int(model.predict(input_frame)[0])
        probability = float(model.predict_proba(input_frame)[0][1]) * 100

        result = {
            "prediction": prediction,
            "probability": round(probability, 2),
            "message": (
                "High risk of heart disease detected"
                if prediction == 1
                else "Low risk of heart disease"
            ),
            "status": "danger" if prediction == 1 else "success",
        }
        save_prediction(request.form, prediction, probability)
        return render_template("result.html", **result)
    except Exception as exc:
        return render_template(
            "result.html",
            message=f"Error: {exc}",
            status="warning",
            prediction=None,
            probability=None,
        )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = normalize_api_payload(request.get_json(silent=True))
        input_frame = build_input_frame(data)
        prediction = int(model.predict(input_frame)[0])
        probability = float(model.predict_proba(input_frame)[0][1])
        if session.get("user_id"):
            save_prediction(data, prediction, probability * 100)

        return jsonify(
            {
                "prediction": prediction,
                "probability": round(probability, 4),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/history")
def history():
    login_redirect = login_required()
    if login_redirect:
        return login_redirect

    with get_db_connection() as connection:
        predictions = connection.execute(
            """
            SELECT
                p.*,
                COALESCE(p.patient_name, u.full_name, 'Unknown') AS display_name,
                COALESCE(p.patient_email, u.email, 'Not provided') AS display_email,
                strftime('%d-%m-%Y', p.created_at) AS saved_date,
                strftime('%H:%M:%S', p.created_at) AS saved_time
            FROM predictions p
            LEFT JOIN users u ON p.user_id = u.id
            WHERE p.user_id = ?
            ORDER BY p.created_at DESC
            """,
            (session["user_id"],),
        ).fetchall()

    return render_template("history.html", predictions=predictions)


@app.route("/all-history")
def all_history():
    login_redirect = login_required()
    if login_redirect:
        return login_redirect

    with get_db_connection() as connection:
        predictions = connection.execute(
            """
            SELECT
                p.*,
                COALESCE(p.patient_name, u.full_name, 'Unknown') AS display_name,
                COALESCE(p.patient_email, u.email, 'Not provided') AS display_email,
                strftime('%d-%m-%Y', p.created_at) AS saved_date,
                strftime('%H:%M:%S', p.created_at) AS saved_time
            FROM predictions p
            LEFT JOIN users u ON p.user_id = u.id
            ORDER BY p.created_at DESC
            """
        ).fetchall()

    return render_template("all_history.html", predictions=predictions)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        Timer(1, open_browser).start()
    app.run(debug=True)
