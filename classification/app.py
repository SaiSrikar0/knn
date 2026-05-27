import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os


st.set_page_config(page_title="Titanic KNN Predictor", layout="wide")


@st.cache_resource
def load_artifacts(models_dir="models"):
	if not os.path.isabs(models_dir):
		models_dir = os.path.join(os.path.dirname(__file__), models_dir)

	model_path = os.path.join(models_dir, "knn_model.pkl")
	scaler_path = os.path.join(models_dir, "scaler.pkl")
	feature_names_path = os.path.join(models_dir, "feature_names.pkl")

	if not os.path.exists(model_path):
		raise FileNotFoundError(f"Model not found: {model_path}")
	if not os.path.exists(scaler_path):
		raise FileNotFoundError(f"Scaler not found: {scaler_path}")
	if not os.path.exists(feature_names_path):
		raise FileNotFoundError(f"Feature names not found: {feature_names_path}")

	with open(model_path, "rb") as f:
		model = pickle.load(f)
	with open(scaler_path, "rb") as f:
		scaler = pickle.load(f)
	with open(feature_names_path, "rb") as f:
		feature_names = pickle.load(f)

	return model, scaler, feature_names


def build_input_df(age, fare, sex, sibsp, parch, pclass, embarked, feature_names):
	# Build a feature vector using the raw numeric codes present in the dataset.
	sex_map = {"male": 0, "female": 1}
	embarked_map = {"S": 0, "C": 1, "Q": 2}
	row = {
		"Age": float(age),
		"Fare": float(fare),
		"Sex": sex_map.get(str(sex).lower(), 0),
		"SibSp": int(sibsp),
		"Parch": int(parch),
		"Pclass": int(pclass),
		"Embarked": embarked_map.get(str(embarked).upper(), 0),
	}
	return pd.DataFrame([row], columns=feature_names)


def main():
	st.title("Titanic Survival Predictor (KNN)")
	st.markdown("Use the controls to enter passenger details and get a survival prediction.")

	# Try to load artifacts
	try:
		model, scaler, feature_names = load_artifacts()
	except Exception as e:
		st.error(f"Error loading model artifacts: {e}")
		st.info("Run the training notebook to generate model artifacts and place them in the `models/` folder.")
		return

	# Preset examples
	presets = {
		"Default": {"Age": 30.0, "Fare": 32.0, "Sex": "male", "SibSp": 0, "Parch": 0, "Pclass": 3, "Embarked": "S"},
		"Young Female 1st": {"Age": 22.0, "Fare": 200.0, "Sex": "female", "SibSp": 0, "Parch": 0, "Pclass": 1, "Embarked": "C"},
		"Elderly Male 3rd": {"Age": 70.0, "Fare": 7.25, "Sex": "male", "SibSp": 0, "Parch": 0, "Pclass": 3, "Embarked": "S"},
		"Child Large Family": {"Age": 5.0, "Fare": 21.0, "Sex": "male", "SibSp": 4, "Parch": 2, "Pclass": 3, "Embarked": "S"},
		"Survived Example (P32)": {"Age": 28.0, "Fare": 146.5208, "Sex": "female", "SibSp": 1, "Parch": 0, "Pclass": 1, "Embarked": "S"}
	}

	col1, col2 = st.columns([1, 1])

	with col1:
		st.header("Passenger Details")
		preset_choice = st.selectbox("Use Preset", options=list(presets.keys()))
		preset = presets[preset_choice]

		age = st.slider("Age", min_value=0.0, max_value=90.0, value=float(preset["Age"]), step=0.5)
		fare = st.number_input("Fare", min_value=0.0, max_value=1000.0, value=float(preset["Fare"]), step=0.1)
		sex = st.selectbox("Sex", options=["male", "female"], index=0 if preset["Sex"] == "male" else 1)
		sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=int(preset["SibSp"]), step=1)
		parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=int(preset["Parch"]), step=1)
		pclass = st.selectbox("Pclass", options=[1, 2, 3], index=[1, 2, 3].index(preset["Pclass"]))
		embarked = st.selectbox("Embarked", options=["S", "C", "Q"], index=["S", "C", "Q"].index(preset["Embarked"]))

		if st.button("Predict"):
			input_df = build_input_df(age, fare, sex, sibsp, parch, pclass, embarked, feature_names)
			X_scaled = scaler.transform(input_df)
			proba = model.predict_proba(X_scaled)[0]
			pred = int(model.predict(X_scaled)[0])

			# Update results in right column
			with col2:
				st.header("Prediction Result")
				surv_pct = float(proba[1]) * 100
				st.metric("Survival Probability", f"{surv_pct:.1f}%")

				st.subheader("Outcome")
				if pred == 1:
					st.success("Model prediction: Survived")
				else:
					st.error("Model prediction: Did not survive")

				st.subheader("Probabilities")
				st.write(f"Death: {proba[0]:.3f}")
				st.write(f"Survive: {proba[1]:.3f}")

				st.subheader("Input features")
				st.table(input_df)

				st.subheader("Model parameters")
				st.write(model.get_params())

	st.markdown("---")
	with st.expander("About this app"):
		st.markdown("This is a simple KNN classifier demo. Enter passenger details and click Predict. The model and scaler are loaded from the `models/` folder.")
		st.markdown("If artifacts are missing run the notebook `notebooks/knn_simple.ipynb` to train and save artifacts to the `models/` folder.")


if __name__ == "__main__":
	main()

