"""
BEH Plomberie Chauffage — Simulateur de devis v2
=================================================
Fonctionnalités :
  1. 🔍 Simulateur          — prédiction unitaire + explication SHAP
  2. ⚖️  Comparateur         — deux devis côte à côte
  3. 🚨 Alertes             — devis en cours à risque
  4. 📊 Dashboard           — graphiques interactifs sur l'historique
  5. 📋 Historique          — log de toutes les simulations
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="BEH — Prédiction de devis",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background: #FAFAFA; }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
      background: #F3F4F6; border-radius: 8px 8px 0 0;
      padding: 8px 18px; font-weight: 600; color: #6B7280;
  }
  .stTabs [aria-selected="true"] {
      background: #6B21A8 !important; color: white !important;
  }
  .metric-card {
      background: white; border-radius: 12px; padding: 18px;
      border-left: 4px solid #6B21A8;
      box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 10px;
  }
  .alert-card {
      background: #FEF2F2; border-radius: 10px; padding: 14px;
      border-left: 4px solid #DC2626; margin-bottom: 8px;
  }
  .ok-card {
      background: #F0FDF4; border-radius: 10px; padding: 14px;
      border-left: 4px solid #16A34A; margin-bottom: 8px;
  }
  .header-bar {
      background: linear-gradient(90deg, #1E1B4B, #6B21A8);
      color: white; padding: 18px 28px; border-radius: 12px;
      margin-bottom: 22px;
  }
</style>
""", unsafe_allow_html=True)

# ── Modèle ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model_beh.pkl", "rb") as f:
        return pickle.load(f)

try:
    model_data = load_model()
    model   = model_data["model"]
    scaler  = model_data["scaler"]
    features = model_data["features"]
    MODEL_OK = True
except Exception:
    MODEL_OK = False

# ── Historique (fichier JSON local) ──────────────────────
HISTORY_FILE = "historique_predictions.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(records):
    with open(HISTORY_FILE, "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def add_to_history(record):
    h = load_history()
    h.append(record)
    save_history(h)

# ── Helper : construire vecteur input ─────────────────────
TYPE_CLIENT_OPTS = ["Particulier", "Cabinet architecte", "Grande entreprise"]
TYPE_PRESTA_OPTS = [
    "Depannage", "Installation chauffage", "Renovation plomberie",
    "Installation climatisation", "Maintenance contrat", "Grosse installation neuve"
]
SAISON_OPTS = ["Hiver", "Printemps", "Ete", "Automne"]

def build_input(type_client, type_prestation, montant, delai, saison, recurrent):
    d = {f: 0 for f in features}
    d["montant_eur"]         = montant
    d["delai_reponse_jours"] = delai
    d["client_recurrent"]    = recurrent

    if type_client == "Grande entreprise":
        d["type_client_Grande entreprise"] = 1
    elif type_client == "Particulier":
        d["type_client_Particulier"] = 1

    presta_map = {
        "Grosse installation neuve":  "type_prestation_Grosse installation neuve",
        "Installation chauffage":     "type_prestation_Installation chauffage",
        "Installation climatisation": "type_prestation_Installation climatisation",
        "Maintenance contrat":        "type_prestation_Maintenance contrat",
        "Renovation plomberie":       "type_prestation_Renovation plomberie",
    }
    if type_prestation in presta_map:
        d[presta_map[type_prestation]] = 1

    saison_map = {"Ete": "saison_Ete", "Hiver": "saison_Hiver", "Printemps": "saison_Printemps"}
    if saison in saison_map:
        d[saison_map[saison]] = 1

    return pd.DataFrame([d])[features]

def predict(df_input):
    scaled = scaler.transform(df_input)
    proba  = model.predict_proba(scaled)[0]
    return proba[1], proba[0]   # (p_accepte, p_refuse)

def verdict(p_acc):
    if p_acc >= 0.70:
        return "✅ Très probablement accepté", "success"
    elif p_acc >= 0.50:
        return "🟡 Probablement accepté — à surveiller", "warning"
    else:
        return "❌ Risque élevé de refus", "error"

# ── SHAP waterfall (matplotlib) ───────────────────────────
def shap_bar_chart(df_input):
    """Simple bar chart des contributions logistiques (approximation SHAP)."""
    try:
        import shap
        explainer = shap.LinearExplainer(model, scaler.transform(df_input))
        sv = explainer.shap_values(scaler.transform(df_input))
        vals = sv[0]
        names = features
    except Exception:
        # Fallback : utiliser les coefficients × valeurs standardisées
        x_s = scaler.transform(df_input)[0]
        vals = model.coef_[0] * x_s
        names = features

    # Garder les 8 plus importantes (en valeur absolue)
    idx  = np.argsort(np.abs(vals))[-8:]
    top_vals  = vals[idx]
    top_names = [names[i].replace("type_prestation_", "").replace("type_client_", "").replace("saison_", "Saison ") for i in idx]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#16A34A" if v > 0 else "#DC2626" for v in top_vals]
    bars = ax.barh(top_names, top_vals, color=colors, height=0.55)
    ax.axvline(0, color="#6B7280", linewidth=0.8)
    ax.set_xlabel("Contribution à la probabilité d'acceptation", fontsize=9)
    ax.set_title("Pourquoi cette prédiction ?", fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(labelsize=9)
    for bar, val in zip(bars, top_vals):
        ax.text(val + (0.003 if val >= 0 else -0.003), bar.get_y() + bar.get_height()/2,
                f"{val:+.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=8)
    green_p = mpatches.Patch(color="#16A34A", label="Favorise l'acceptation")
    red_p   = mpatches.Patch(color="#DC2626", label="Favorise le refus")
    ax.legend(handles=[green_p, red_p], fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
  <span style="font-size:22px;font-weight:700;">🔧 BEH Plomberie Chauffage</span>
  <span style="margin-left:16px;font-size:14px;opacity:0.85;">
      Outil d'aide à la décision — Prédiction d'acceptation des devis
  </span>
</div>
""", unsafe_allow_html=True)

if not MODEL_OK:
    st.error("❌ Fichier model_beh.pkl introuvable. Placez-le dans le même dossier que app.py.")
    st.stop()

# ─────────────────────────────────────────────────────────
# ONGLETS
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Simulateur",
    "⚖️  Comparateur",
    "🚨 Alertes",
    "📊 Dashboard",
    "📋 Historique",
])

# ══════════════════════════════════════════════════════════
# ONGLET 1 — SIMULATEUR
# ══════════════════════════════════════════════════════════
with tab1:
    st.subheader("Simulateur de devis")
    st.caption("Renseignez les caractéristiques du devis pour obtenir une prédiction et une explication.")

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        tc  = st.selectbox("Type de client", TYPE_CLIENT_OPTS, key="s_tc")
        tp  = st.selectbox("Type de prestation", TYPE_PRESTA_OPTS, key="s_tp")
        sa  = st.selectbox("Saison", SAISON_OPTS, key="s_sa")
    with col2:
        mt  = st.number_input("Montant du devis (€)", 100, 100000, 3000, 500, key="s_mt")
        dl  = st.slider("Délai de réponse estimé (jours)", 1, 60, 10, key="s_dl")
        rc  = st.radio("Client récurrent ?", [0, 1], format_func=lambda x: "✅ Oui" if x else "❌ Non",
                       horizontal=True, key="s_rc")

    if st.button("🔍 Prédire", use_container_width=True, type="primary", key="btn_sim"):
        df_in = build_input(tc, tp, mt, dl, sa, rc)
        p_acc, p_ref = predict(df_in)
        label, level = verdict(p_acc)

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Probabilité d'acceptation", f"{p_acc*100:.1f}%")
        c2.metric("Probabilité de refus",       f"{p_ref*100:.1f}%")
        c3.metric("Prédiction",                 "Accepté ✅" if p_acc >= 0.5 else "Refusé ❌")

        st.progress(float(p_acc))

        if level == "success":
            st.success(label)
        elif level == "warning":
            st.warning(label)
        else:
            st.error(label)

        # Conseil contextuel
        conseils = []
        if p_acc < 0.5:
            if dl > 20:
                conseils.append("📅 Le délai de réponse est long (>20j) — relancez rapidement.")
            if mt > 15000:
                conseils.append("💶 Montant élevé — envisagez un argumentaire commercial renforcé.")
            if rc == 0:
                conseils.append("👤 Nouveau client — établissez la confiance avant d'envoyer.")
        if conseils:
            st.info("**Actions recommandées :**\n" + "\n".join(f"- {c}" for c in conseils))

        # Graphique SHAP
        with st.expander("📈 Pourquoi cette prédiction ? (explication des variables)", expanded=True):
            fig = shap_bar_chart(df_in)
            st.pyplot(fig, use_container_width=True)
            st.caption("Vert = facteur favorable à l'acceptation | Rouge = facteur de risque de refus")

        # Sauvegarde historique
        add_to_history({
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "type_client": tc, "type_prestation": tp,
            "montant": mt, "delai": dl, "saison": sa, "recurrent": rc,
            "p_accepte": round(p_acc*100, 1),
            "prediction": "Accepté" if p_acc >= 0.5 else "Refusé"
        })

# ══════════════════════════════════════════════════════════
# ONGLET 2 — COMPARATEUR
# ══════════════════════════════════════════════════════════
with tab2:
    st.subheader("Comparateur de scénarios")
    st.caption("Simulez deux versions d'un même devis pour identifier la meilleure configuration.")

    col_a, col_b = st.columns(2, gap="large")

    def devis_form(col, label, prefix):
        with col:
            st.markdown(f"### {label}")
            tc = st.selectbox("Type de client",     TYPE_CLIENT_OPTS, key=f"{prefix}_tc")
            tp = st.selectbox("Type de prestation", TYPE_PRESTA_OPTS, key=f"{prefix}_tp")
            sa = st.selectbox("Saison",             SAISON_OPTS,      key=f"{prefix}_sa")
            mt = st.number_input("Montant (€)", 100, 100000, 3000, 500, key=f"{prefix}_mt")
            dl = st.slider("Délai (jours)", 1, 60, 10, key=f"{prefix}_dl")
            rc = st.radio("Client récurrent ?", [0, 1],
                          format_func=lambda x: "Oui" if x else "Non",
                          horizontal=True, key=f"{prefix}_rc")
        return tc, tp, sa, mt, dl, rc

    tc_a, tp_a, sa_a, mt_a, dl_a, rc_a = devis_form(col_a, "📄 Devis A", "a")
    tc_b, tp_b, sa_b, mt_b, dl_b, rc_b = devis_form(col_b, "📄 Devis B", "b")

    if st.button("⚖️  Comparer les deux devis", use_container_width=True, type="primary"):
        df_a = build_input(tc_a, tp_a, mt_a, dl_a, sa_a, rc_a)
        df_b = build_input(tc_b, tp_b, mt_b, dl_b, sa_b, rc_b)
        pa, _ = predict(df_a)
        pb, _ = predict(df_b)

        st.markdown("---")
        st.subheader("Résultat de la comparaison")
        r1, r2, r3 = st.columns(3)
        r1.metric("Devis A — probabilité d'acceptation", f"{pa*100:.1f}%")
        r2.metric("Devis B — probabilité d'acceptation", f"{pb*100:.1f}%",
                  delta=f"{(pb-pa)*100:+.1f}% vs A")
        winner = "A ✅" if pa >= pb else "B ✅"
        r3.metric("Meilleure configuration", f"Devis {winner}")

        # Graphique barres comparatif
        fig, ax = plt.subplots(figsize=(5, 2.8))
        ax.barh(["Devis B", "Devis A"], [pb*100, pa*100],
                color=["#6B21A8" if pb > pa else "#D1D5DB",
                       "#6B21A8" if pa >= pb else "#D1D5DB"],
                height=0.45)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probabilité d'acceptation (%)", fontsize=9)
        ax.axvline(50, color="#DC2626", linewidth=1, linestyle="--", label="Seuil 50%")
        for i, v in enumerate([pb*100, pa*100]):
            ax.text(v+1, i, f"{v:.1f}%", va="center", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)

        # Différences clés
        diffs = []
        if mt_a != mt_b:
            diffs.append(f"Montant : {mt_a}€ vs {mt_b}€")
        if dl_a != dl_b:
            diffs.append(f"Délai : {dl_a}j vs {dl_b}j")
        if tc_a != tc_b:
            diffs.append(f"Client : {tc_a} vs {tc_b}")
        if rc_a != rc_b:
            diffs.append(f"Récurrence : {'Oui' if rc_a else 'Non'} vs {'Oui' if rc_b else 'Non'}")
        if diffs:
            st.info("**Différences entre les deux devis :**\n" + "\n".join(f"- {d}" for d in diffs))

# ══════════════════════════════════════════════════════════
# ONGLET 3 — ALERTES
# ══════════════════════════════════════════════════════════
with tab3:
    st.subheader("🚨 Système d'alertes — Devis en cours à risque")
    st.caption("Uploadez un fichier CSV de vos devis en cours pour identifier ceux à risque de refus.")

    with st.expander("📋 Format du fichier CSV attendu", expanded=False):
        st.code("""type_client,type_prestation,montant_eur,delai_reponse_jours,saison,client_recurrent
Particulier,Depannage,800,5,Hiver,1
Grande entreprise,Grosse installation neuve,35000,30,Automne,0
Cabinet architecte,Renovation plomberie,8000,15,Printemps,1""")
        st.caption("Les valeurs de type_client, type_prestation et saison doivent correspondre exactement aux options du simulateur.")

    # Génération d'un exemple téléchargeable
    sample_df = pd.DataFrame({
        "type_client":        ["Particulier","Grande entreprise","Cabinet architecte","Particulier","Grande entreprise"],
        "type_prestation":    ["Depannage","Grosse installation neuve","Renovation plomberie","Installation chauffage","Maintenance contrat"],
        "montant_eur":        [800, 35000, 8000, 4500, 1200],
        "delai_reponse_jours":[5, 30, 15, 8, 25],
        "saison":             ["Hiver","Automne","Printemps","Ete","Hiver"],
        "client_recurrent":   [1, 0, 1, 0, 1],
    })
    st.download_button("⬇️ Télécharger un fichier exemple", sample_df.to_csv(index=False),
                       "exemple_devis.csv", "text/csv")

    uploaded = st.file_uploader("Uploadez votre fichier CSV de devis en cours", type=["csv"])

    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            required = {"type_client","type_prestation","montant_eur","delai_reponse_jours","saison","client_recurrent"}
            if not required.issubset(df_up.columns):
                st.error(f"Colonnes manquantes : {required - set(df_up.columns)}")
            else:
                # Prédiction pour chaque ligne
                results = []
                for _, row in df_up.iterrows():
                    df_r = build_input(
                        row["type_client"], row["type_prestation"],
                        row["montant_eur"], row["delai_reponse_jours"],
                        row["saison"], row["client_recurrent"]
                    )
                    pa, pr = predict(df_r)
                    results.append({**row.to_dict(), "p_accepte_%": round(pa*100,1),
                                    "p_refuse_%": round(pr*100,1),
                                    "alerte": "🚨 RISQUE" if pa < 0.5 else ("⚠️ À surveiller" if pa < 0.70 else "✅ OK")})

                df_res = pd.DataFrame(results).sort_values("p_accepte_%")

                # KPIs
                n_risque    = (df_res["alerte"] == "🚨 RISQUE").sum()
                n_surveiller= (df_res["alerte"] == "⚠️ À surveiller").sum()
                n_ok        = (df_res["alerte"] == "✅ OK").sum()

                k1, k2, k3 = st.columns(3)
                k1.metric("🚨 Devis à risque élevé",   n_risque)
                k2.metric("⚠️ À surveiller",            n_surveiller)
                k3.metric("✅ En bonne voie",            n_ok)

                st.markdown("---")
                st.subheader("Détail par devis")

                for _, row in df_res.iterrows():
                    card_class = "alert-card" if row["alerte"] == "🚨 RISQUE" else (
                                 "metric-card" if row["alerte"] == "⚠️ À surveiller" else "ok-card")
                    st.markdown(f"""
                    <div class="{card_class}">
                      <b>{row['alerte']}</b> &nbsp;|&nbsp;
                      {row['type_client']} — {row['type_prestation']} — {row['montant_eur']}€ — {row['delai_reponse_jours']}j
                      &nbsp;|&nbsp; Proba acceptation : <b>{row['p_accepte_%']}%</b>
                    </div>
                    """, unsafe_allow_html=True)

                # Export résultats
                st.download_button("⬇️ Télécharger les résultats",
                                   df_res.to_csv(index=False),
                                   "alertes_devis.csv", "text/csv")
        except Exception as e:
            st.error(f"Erreur lors du traitement : {e}")

    else:
        st.info("Uploadez un fichier CSV pour voir les alertes. Vous pouvez télécharger le fichier exemple ci-dessus pour tester.")

# ══════════════════════════════════════════════════════════
# ONGLET 4 — DASHBOARD
# ══════════════════════════════════════════════════════════
with tab4:
    st.subheader("📊 Dashboard analytique")
    st.caption("Explorez les tendances du dataset de devis BEH.")

    @st.cache_data
    def load_dataset():
        try:
            return pd.read_csv("devis_beh_dataset.csv")
        except FileNotFoundError:
            return None

    df_dash = load_dataset()

    if df_dash is None:
        st.warning("Fichier devis_beh_dataset.csv introuvable. Placez-le dans le même dossier que app.py.")
    else:
        # KPIs globaux
        total    = len(df_dash)
        acceptes = (df_dash["statut"] == "Accepte").sum()
        refuses  = (df_dash["statut"] == "Refuse").sum()
        taux     = acceptes / total * 100

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total devis",         total)
        k2.metric("Acceptés",            acceptes)
        k3.metric("Refusés",             refuses)
        k4.metric("Taux d'acceptation",  f"{taux:.1f}%")

        st.markdown("---")

        # Filtres
        fc1, fc2 = st.columns(2)
        filtre_client = fc1.multiselect("Filtrer par type de client",
                                        df_dash["type_client"].unique().tolist(),
                                        default=df_dash["type_client"].unique().tolist())
        filtre_statut = fc2.radio("Statut", ["Tous", "Accepté", "Refusé"], horizontal=True)

        df_f = df_dash[df_dash["type_client"].isin(filtre_client)]
        if filtre_statut == "Accepté":
            df_f = df_f[df_f["statut"] == "Accepte"]
        elif filtre_statut == "Refusé":
            df_f = df_f[df_f["statut"] == "Refuse"]

        # Graphiques
        g1, g2 = st.columns(2)

        with g1:
            st.markdown("**Taux d'acceptation par type de client**")
            tx = df_f.groupby("type_client")["statut"].apply(
                lambda s: (s == "Accepte").mean() * 100).reset_index()
            tx.columns = ["Type client", "Taux (%)"]
            fig, ax = plt.subplots(figsize=(5, 3))
            bars = ax.bar(tx["Type client"], tx["Taux (%)"],
                          color=["#6B21A8","#EA580C","#16A34A"][:len(tx)], width=0.5)
            ax.set_ylim(0, 100)
            ax.axhline(50, color="#DC2626", linewidth=1, linestyle="--", alpha=0.6)
            ax.set_ylabel("Taux d'acceptation (%)", fontsize=9)
            for bar, val in zip(bars, tx["Taux (%)"]):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                        f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
            ax.tick_params(labelsize=8)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

        with g2:
            st.markdown("**Distribution des montants (Accepté vs Refusé)**")
            acc = df_f[df_f["statut"]=="Accepte"]["montant_eur"]
            ref = df_f[df_f["statut"]=="Refuse"]["montant_eur"]
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(acc, bins=20, alpha=0.65, color="#16A34A", label="Accepté", density=True)
            ax.hist(ref, bins=20, alpha=0.65, color="#DC2626", label="Refusé",  density=True)
            ax.set_xlabel("Montant (€)", fontsize=9)
            ax.set_ylabel("Densité", fontsize=9)
            ax.legend(fontsize=8)
            ax.tick_params(labelsize=8)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

        g3, g4 = st.columns(2)

        with g3:
            st.markdown("**Taux d'acceptation par saison**")
            ts = df_f.groupby("saison")["statut"].apply(
                lambda s: (s == "Accepte").mean() * 100).reset_index()
            ts.columns = ["Saison", "Taux (%)"]
            order = ["Hiver", "Printemps", "Ete", "Automne"]
            ts = ts.set_index("Saison").reindex(order).reset_index()
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(ts["Saison"], ts["Taux (%)"], marker="o", color="#6B21A8",
                    linewidth=2, markersize=7)
            ax.fill_between(ts["Saison"], ts["Taux (%)"], alpha=0.15, color="#6B21A8")
            ax.set_ylim(0, 100)
            ax.set_ylabel("Taux d'acceptation (%)", fontsize=9)
            ax.tick_params(labelsize=8)
            for x, y in zip(ts["Saison"], ts["Taux (%)"]):
                if pd.notna(y):
                    ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                                xytext=(0, 8), ha="center", fontsize=8)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

        with g4:
            st.markdown("**Délai de réponse moyen (Accepté vs Refusé)**")
            dl_stats = df_f.groupby("statut")["delai_reponse_jours"].mean().reset_index()
            dl_stats["statut"] = dl_stats["statut"].map({"Accepte":"Accepté","Refuse":"Refusé"})
            fig, ax = plt.subplots(figsize=(5, 3))
            colors = ["#16A34A" if s == "Accepté" else "#DC2626" for s in dl_stats["statut"]]
            bars = ax.bar(dl_stats["statut"], dl_stats["delai_reponse_jours"],
                          color=colors, width=0.4)
            ax.set_ylabel("Délai moyen (jours)", fontsize=9)
            for bar, val in zip(bars, dl_stats["delai_reponse_jours"]):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                        f"{val:.1f}j", ha="center", fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=9)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("**Taux d'acceptation par type de prestation**")
        tp_stats = df_f.groupby("type_prestation")["statut"].apply(
            lambda s: (s == "Accepte").mean() * 100).sort_values(ascending=True).reset_index()
        tp_stats.columns = ["Prestation", "Taux (%)"]
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.barh(tp_stats["Prestation"], tp_stats["Taux (%)"], color="#6B21A8", height=0.5)
        ax.set_xlim(0, 100)
        ax.axvline(50, color="#DC2626", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_xlabel("Taux d'acceptation (%)", fontsize=9)
        for i, val in enumerate(tp_stats["Taux (%)"]):
            ax.text(val+1, i, f"{val:.1f}%", va="center", fontsize=9)
        ax.tick_params(labelsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)

# ══════════════════════════════════════════════════════════
# ONGLET 5 — HISTORIQUE
# ══════════════════════════════════════════════════════════
with tab5:
    st.subheader("📋 Historique des prédictions")
    st.caption("Toutes les simulations effectuées dans l'onglet Simulateur sont enregistrées ici.")

    history = load_history()

    if not history:
        st.info("Aucune simulation effectuée pour l'instant. Utilisez l'onglet Simulateur pour commencer.")
    else:
        df_hist = pd.DataFrame(history)

        # KPIs historique
        h1, h2, h3 = st.columns(3)
        h1.metric("Total simulations",      len(df_hist))
        h2.metric("Devis prédits Acceptés", (df_hist["prediction"]=="Accepté").sum())
        h3.metric("Devis prédits Refusés",  (df_hist["prediction"]=="Refusé").sum())

        # Tableau
        st.markdown("---")
        df_display = df_hist.copy()
        df_display["Résultat"] = df_display["prediction"].map(
            {"Accepté": "✅ Accepté", "Refusé": "❌ Refusé"})
        df_display = df_display.rename(columns={
            "date":             "Date",
            "type_client":      "Client",
            "type_prestation":  "Prestation",
            "montant":          "Montant (€)",
            "delai":            "Délai (j)",
            "p_accepte":        "Proba acceptation (%)",
        })

        cols_show = ["Date","Client","Prestation","Montant (€)","Délai (j)","Proba acceptation (%)","Résultat"]
        st.dataframe(df_display[cols_show].sort_values("Date", ascending=False),
                     use_container_width=True, hide_index=True)

        col_dl, col_cl = st.columns([1, 1])
        col_dl.download_button("⬇️ Exporter l'historique (CSV)",
                               df_display[cols_show].to_csv(index=False),
                               "historique_predictions.csv", "text/csv")
        if col_cl.button("🗑️ Effacer l'historique"):
            save_history([])
            st.rerun()

# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.caption("BEH Plomberie Chauffage — Projet Data Science | Master Big Data & AI | ÉSTIAM 2025-2026 | "
           "[GitHub](https://github.com/KGsaccount/beh-devis-prediction)")
