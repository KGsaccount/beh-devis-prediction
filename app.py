import streamlit as st
import pandas as pd
import numpy as np
import pickle
 
# Chargement du modèle
@st.cache_resource
def load_model():
    with open("model_beh.pkl", "rb") as f:
        return pickle.load(f)
 
model_data = load_model()
model   = model_data["model"]
scaler  = model_data["scaler"]
features = model_data["features"]
 
# -------------------------------------------------------
# Interface
# -------------------------------------------------------
st.set_page_config(page_title="BEH — Prédiction de devis", page_icon="🔧", layout="centered")
 
st.title("🔧 BEH Plomberie Chauffage")
st.subheader("Simulateur de prédiction d'acceptation de devis")
st.markdown("---")
st.markdown(
    "Renseignez les caractéristiques du devis ci-dessous. "
    "Le modèle estime la probabilité que le client l'accepte."
)
 
# -------------------------------------------------------
# Formulaire de saisie
# -------------------------------------------------------
col1, col2 = st.columns(2)
 
with col1:
    type_client = st.selectbox(
        "Type de client",
        ["Particulier", "Cabinet architecte", "Grande entreprise"]
    )
    type_prestation = st.selectbox(
        "Type de prestation",
        [
            "Depannage",
            "Installation chauffage",
            "Renovation plomberie",
            "Installation climatisation",
            "Maintenance contrat",
            "Grosse installation neuve",
        ]
    )
    saison = st.selectbox("Saison", ["Hiver", "Printemps", "Ete", "Automne"])
 
with col2:
    montant = st.number_input(
        "Montant du devis (€)", min_value=100, max_value=100000,
        value=3000, step=100
    )
    delai = st.slider(
        "Délai de réponse du client (jours)", min_value=1, max_value=60, value=10
    )
    client_recurrent = st.radio(
        "Client récurrent ?", options=[0, 1],
        format_func=lambda x: "Oui" if x == 1 else "Non",
        horizontal=True
    )
 
st.markdown("---")
 
# -------------------------------------------------------
# Prédiction
# -------------------------------------------------------
if st.button("🔍 Prédire l'acceptation du devis", use_container_width=True):
 
    # Construire le vecteur d'entrée avec les mêmes colonnes que l'entraînement
    input_dict = {f: 0 for f in features}
    input_dict["montant_eur"]          = montant
    input_dict["delai_reponse_jours"]  = delai
    input_dict["client_recurrent"]     = client_recurrent
 
    # One-hot encoding manuel (même logique que pd.get_dummies drop_first=True)
    # type_client : référence = Cabinet architecte
    if type_client == "Grande entreprise":
        input_dict["type_client_Grande entreprise"] = 1
    elif type_client == "Particulier":
        input_dict["type_client_Particulier"] = 1
 
    # type_prestation : référence = Depannage
    if type_prestation == "Grosse installation neuve":
        input_dict["type_prestation_Grosse installation neuve"] = 1
    elif type_prestation == "Installation chauffage":
        input_dict["type_prestation_Installation chauffage"] = 1
    elif type_prestation == "Installation climatisation":
        input_dict["type_prestation_Installation climatisation"] = 1
    elif type_prestation == "Maintenance contrat":
        input_dict["type_prestation_Maintenance contrat"] = 1
    elif type_prestation == "Renovation plomberie":
        input_dict["type_prestation_Renovation plomberie"] = 1
 
    # saison : référence = Automne
    if saison == "Ete":
        input_dict["saison_Ete"] = 1
    elif saison == "Hiver":
        input_dict["saison_Hiver"] = 1
    elif saison == "Printemps":
        input_dict["saison_Printemps"] = 1
 
    input_df = pd.DataFrame([input_dict])[features]
    input_scaled = scaler.transform(input_df)
 
    proba = model.predict_proba(input_scaled)[0]
    proba_accepte = proba[1]
    proba_refuse  = proba[0]
    prediction    = "Accepté ✅" if proba_accepte >= 0.5 else "Refusé ❌"
 
    # Affichage résultat
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        if proba_accepte >= 0.5:
            st.success(f"### Prédiction : {prediction}")
        else:
            st.error(f"### Prédiction : {prediction}")
    with col_res2:
        st.metric("Probabilité d'acceptation", f"{proba_accepte*100:.1f}%")
        st.metric("Probabilité de refus",       f"{proba_refuse*100:.1f}%")
 
    # Jauge de confiance
    st.progress(float(proba_accepte), text=f"Score de confiance : {proba_accepte*100:.1f}%")
 
    # Interprétation en langage clair
    st.markdown("---")
    st.markdown("#### 💡 Interprétation")
    if proba_accepte >= 0.75:
        st.info("Ce devis a de très bonnes chances d'être accepté. Pas d'action particulière requise.")
    elif proba_accepte >= 0.5:
        st.warning(
            "Ce devis a des chances d'être accepté, mais reste incertain. "
            "Envisagez un suivi rapide ou une relance sous 48h."
        )
    else:
        st.error(
            "Ce devis est à risque de refus. Facteurs défavorables possibles : "
            "montant élevé, délai de réponse long, nouveau client. "
            "Envisagez un ajustement ou une relance proactive."
        )
 
st.markdown("---")
st.caption("Projet Data — BEH Plomberie Chauffage | Master Big Data & AI | ÉSTIAM 2025-2026")
