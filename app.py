import streamlit as st
import pickle

# Cargar el modelo y el DictVectorizer
with open('churn-model.pck', 'rb') as f:
    dv, model = pickle.load(f)

st.title("Predicción de Churn de Clientes (v2)")
st.header("Introduce los 11 datos clave del cliente:")

# 11 atributos importantes
contract = st.selectbox("Tipo de contrato", ["month-to-month", "one_year", "two_year"])
internet_service = st.selectbox("Tipo de Internet", ["dsl", "fiber_optic", "no"])
online_security = st.selectbox("Seguridad online", ["no", "yes", "no_internet_service"])
tech_support = st.selectbox("Soporte técnico", ["no", "yes", "no_internet_service"])
online_backup = st.selectbox("Copia de seguridad online", ["no", "yes", "no_internet_service"])
device_protection = st.selectbox("Protección de dispositivos", ["no", "yes", "no_internet_service"])
payment_method = st.selectbox("Método de pago", [
    "bank_transfer_(automatic)",
    "credit_card_(automatic)",
    "electronic_check",
    "mailed_check"
])
paperless_billing = st.selectbox("Facturación sin papel", ["no", "yes"])
tenure = st.number_input("Antigüedad (meses)", min_value=0, max_value=100, step=1)
monthly_charges = st.number_input("Cargos mensuales", min_value=0.0, max_value=500.0, step=0.1)
total_charges = st.number_input("Cargos totales", min_value=0.0, max_value=10000.0, step=0.1)

if st.button("Predecir"):
    client_data = {
        "contract": contract,
        "internetservice": internet_service,
        "onlinesecurity": online_security,
        "techsupport": tech_support,
        "onlinebackup": online_backup,
        "deviceprotection": device_protection,
        "paymentmethod": payment_method,
        "paperlessbilling": paperless_billing,
        "tenure": tenure,
        "monthlycharges": monthly_charges,
        "totalcharges": total_charges
    }

    X_client = dv.transform([client_data])
    y_pred_proba = model.predict_proba(X_client)[0][1]

    st.subheader("Resultado:")
    if y_pred_proba > 0.5:
        st.error(f"El cliente tiene una alta probabilidad de churn: {y_pred_proba:.2f}")
    else:
        st.success(f"El cliente tiene una baja probabilidad de churn: {y_pred_proba:.2f}")