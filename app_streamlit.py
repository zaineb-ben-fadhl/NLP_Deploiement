import streamlit as st
import requests
import json

API_URL = "https://dreaddit-api.yellowmeadow-d89419e0.francecentral.azurecontainerapps.io/predict"

st.set_page_config(page_title="Dreaddit Stress Detector", page_icon="🧠")

st.title("Stress Detection ")
st.write("Analyse automatique du stress à partir d’un texte Reddit")

user_text = st.text_area("Entrez un texte :", height=150)

if st.button("🔍 Analyser"):
    if not user_text.strip():
        st.warning("⚠️ Veuillez entrer un texte.")
    else:
        with st.spinner("Analyse en cours..."):
            response = requests.post(
                API_URL,
                headers={"Content-Type": "application/json"},
                data=json.dumps({"text": user_text}),
                timeout=15
            )

            st.caption(f"HTTP status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()

                st.subheader(" Réponse brute du modèle")
                st.json(result)

                prediction = result.get("prediction")
                label_text = result.get("label")
                probability = result.get("probability")
                risk_level = result.get("risk_level")

                st.subheader("Résultat du modèle")

                if prediction == 1:
                    st.error(f"😰 **Stress détecté**")
                elif prediction == 0:
                    st.success(f"😌 **Pas de stress détecté**")

                if probability is not None:
                    st.write(f"**Probabilité de stress :** {probability:.2%}")

                if risk_level:
                    st.write(f"**Niveau de risque :** {risk_level}")
            else:
                st.error(f"Erreur API ({response.status_code})")
                st.text(response.text)
