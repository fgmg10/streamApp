import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Configuración de la interfaz de usuario
st.title('Selección de Algoritmo de Machine Learning')
st.sidebar.header('Opciones')

# Cargar el archivo CSV desde el cuadro de diálogo
uploaded_file = st.sidebar.file_uploader("Elige un archivo CSV", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(f'Dataset cargado: {uploaded_file.name}')
    st.dataframe(data.head())
    
    # Mostrar figura de la correlación de datos
    if st.sidebar.button('Mostrar figura de correlación'):
        st.write('Figura de correlación de datos:')
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
        st.pyplot(plt)

    # Selección del mejor algoritmo de machine learning
    st.sidebar.header('Seleccionar el objetivo (target)')
    target = st.sidebar.selectbox('Variable objetivo', data.columns)

    if target:
        X = data.drop(columns=[target])
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        def evaluate_models(X_train, X_test, y_train, y_test):
            models = {
                "Random Forest": RandomForestClassifier(),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
                "Logistic Regression": LogisticRegression(),
                "Support Vector Classifier": SVC(probability=True),  # Añadir probability=True
                "Decision Tree": DecisionTreeClassifier(),
                "Naive Bayes": GaussianNB(),
                "Stochastic Gradient Descent": SGDClassifier(),
                "K Nearest Neighbor": KNeighborsClassifier()
            }

            results = {}
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    results[name] = accuracy
                except ValueError as e:
                    st.warning(f"Error al entrenar el modelo {name}: {e}")
                    results[name] = None

            return results

        # Botón para seleccionar el mejor algoritmo
        if st.sidebar.button('Seleccionar el mejor algoritmo'):
            results = evaluate_models(X_train, X_test, y_train, y_test)
            # Filtrar resultados válidos
            valid_results = {k: v for k, v in results.items() if v is not None}
            if valid_results:
                best_model_name = max(valid_results, key=valid_results.get)
                best_model = {
                    "Random Forest": RandomForestClassifier(),
                    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
                    "Logistic Regression": LogisticRegression(),
                    "Support Vector Classifier": SVC(probability=True),  # Añadir probability=True
                    "Decision Tree": DecisionTreeClassifier(),
                    "Naive Bayes": GaussianNB(),
                    "Stochastic Gradient Descent": SGDClassifier(),
                    "K Nearest Neighbor": KNeighborsClassifier()
                }[best_model_name]
                best_model.fit(X_train, y_train)
                st.session_state['best_model'] = best_model
                st.session_state['scaler'] = scaler
                st.session_state['best_model_name'] = best_model_name
                st.write(f'El mejor modelo es: {best_model_name} con una precisión de {valid_results[best_model_name]:.2f}')
                st.bar_chart(valid_results)
            else:
                st.write('Ningún modelo se entrenó correctamente.')

        # Mostrar el formulario de predicción solo si hay un modelo entrenado
        if 'best_model' in st.session_state:
            st.sidebar.header('Formulario de predicción')
            prediction_input = {}
            for col in X.columns:
                prediction_input[col] = st.sidebar.number_input(f'Valor para {col}', value=float(X[col].mean()))
            
            if st.sidebar.button('Predecir'):
                input_df = pd.DataFrame([prediction_input])
                input_scaled = st.session_state['scaler'].transform(input_df)
                
                best_model = st.session_state['best_model']
                
                if hasattr(best_model, "predict_proba"):
                    prediction_proba = best_model.predict_proba(input_scaled)
                    st.sidebar.write(f'Probabilidad de tener diabetes: {prediction_proba[0][1]:.2f}')
                else:
                    prediction = best_model.predict(input_scaled)
                    st.sidebar.write(f'Predicción de la variable objetivo: {prediction[0]}')
