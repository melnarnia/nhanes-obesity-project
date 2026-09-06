import streamlit as st
import pandas as pd
import plotly.express as px
import time
import joblib
import numpy as np

# Page Config
st.set_page_config(page_title="Adipositas Dashboard", page_icon="🍔")

@st.cache_data
def load_data():
    return pd.read_csv("obesity_cleaned.csv")

# ML model
@st.cache_resource
def load_model():
    return joblib.load("obesity_model.pkl")

@st.cache_resource
def load_medians():
    return joblib.load("training_medians.pkl")

def main():
    # Titel
    st.title("Adipositas Dashboard 🍔")
    st.write("Interaktive Datenanalyse von demografischen und Lifestyle Daten aus NHANES (National Health and Nutrition Examination Survey) 08/2021 - 08/2023. Dabei handelt es sich um eine Erhebung, bei der Daten zu Gesundheit und Ernährung von ausgewählten Bürgern in den USA erhoben werden. Diese Analyse fokussiert sich auf Daten zu Adipositas und beeinflussende Faktoren.")

    # Daten laden
    with st.spinner("Lade Daten..."):
        time.sleep(2)
        df = load_data()
    
    # Sidebar: Filter
    st.sidebar.title("Filter ⚙️")
    age_range = st.sidebar.slider(
        "Alter:",
        int(df["age"].min()),
        int(df["age"].max()),
        (20, 80)
    )
    
    bmi_options = {
        "Normalgewichtig": 0,
        "Übergewichtig": 1,
        "Adipös": 2
    }
    show_bmi = st.sidebar.multiselect(
        "BMI-Kategorie(n):",
        list(bmi_options.keys()),
        default=list(bmi_options.keys())
    )

    gender_choice = st.sidebar.radio(
        "Geschlecht:",
        ["Alle", "Männer", "Frauen"],
        horizontal=True
    )
    
    # Daten filtern
    filtered_df = df[
        (df["age"] >= age_range[0]) & 
        (df["age"] <= age_range[1])
    ]
    
    selected_codes = [bmi_options[label] for label in show_bmi]
    filtered_df = filtered_df[filtered_df["bmi_cat"].isin(selected_codes)]

    if gender_choice == "Männer":
        filtered_df = filtered_df[filtered_df["gender"] == 1.0]
    elif gender_choice == "Frauen":
        filtered_df = filtered_df[filtered_df["gender"] == 2.0]

    # Definition BMI
    bmi_order = [0, 1, 2]
    bmi_labels = ["Normalgewichtig", "Übergewichtig", "Adipös"]
    bmi_colors = ["green", "orange", "red"]

    # Feature engineering
    def engineer_features(df, medians):
        df = df.copy()
        df["sleep_sedentary_interaction"] = df["sleep_avg"] * df["sedentary_act_min"]
        df["alcohol_sedentary_interaction"] = df["alcohol"] * df["sedentary_act_min"]
        df["activity_sedentary_ratio"] = df["activity_level"] / (df["sedentary_act_min"] + 1)
        df["sugar_fiber_ratio"] = df["sugar"] / (df["fiber"] + 1)
        df["protein_per_kcal"] = df["protein"] / (df["kcal"] + 1)
        df["age_group_young"] = (df["age"] < 35).astype(int)
        df["age_group_middle"] = ((df["age"] >= 35) & (df["age"] < 60)).astype(int)
        df["age_group_old"] = (df["age"] >= 60).astype(int)
        df["ses_score"] = df["income"] + df["education_lvl"]
        df["lifestyle_risk_score"] = (
            (df["sedentary_act_min"] > medians["sedentary_act_min"]).astype(int) +
            (df["activity_level"] < medians["activity_level"]).astype(int) +
            df["smoke_status"] +
            (df["alcohol"] > medians["alcohol"]).astype(int)
        )
        return df

    REQUIRED_COLUMNS = [
        "gender", "age", "education_lvl", "income", "kcal", "protein", "carbs",
        "sugar", "fiber", "fat", "alcohol", "sedentary_act_min", "smoke_status",
        "activity_level", "sleep_avg"
    ]
    
    COLUMN_ORDER = [
        "gender", "age", "education_lvl", "income", "kcal", "protein", "carbs",
        "sugar", "fiber", "fat", "alcohol", "sedentary_act_min", "smoke_status",
        "activity_level", "sleep_avg", "sleep_sedentary_interaction",
        "alcohol_sedentary_interaction", "activity_sedentary_ratio",
        "sugar_fiber_ratio", "protein_per_kcal", "age_group_young",
        "age_group_middle", "age_group_old", "ses_score", "lifestyle_risk_score"
    ]

    # deutsche sprechende Labels
    feature_labels_de = {
        "gender": "Geschlecht",
        "age": "Alter",
        "education_lvl": "Bildungsniveau",
        "income": "Einkommen",
        "kcal": "Kalorien",
        "protein": "Protein",
        "carbs": "Kohlenhydrate",
        "sugar": "Zucker",
        "fiber": "Ballaststoffe",
        "fat": "Fett",
        "alcohol": "Alkoholkonsum",
        "sedentary_act_min": "Inaktive Minuten/Tag",
        "smoke_status": "Rauchstatus",
        "activity_level": "Aktivitätslevel",
        "bmi_cat": "BMI-Kategorie",
        "sleep_avg": "Ø Schlafdauer",
        "sleep_sedentary_interaction": "Schlaf × Inaktivität",
        "alcohol_sedentary_interaction": "Alkohol × Inaktivität",
        "activity_sedentary_ratio": "Aktivität/Inaktivität-Verhältnis",
        "sugar_fiber_ratio": "Zucker/Ballaststoffe-Verhältnis",
        "protein_per_kcal": "Protein pro Kalorie",
        "age_group_young": "Altersgruppe: jung (<35)",
        "age_group_middle": "Altersgruppe: mittel (35–59)",
        "age_group_old": "Altersgruppe: alt (60+)",
        "ses_score": "Sozioökonomischer Score",
        "lifestyle_risk_score": "Lifestyle-Risiko-Score"
    }

    # Einheiten für metrische Variablen (nur dort ergänzt, wo die Einheit
    # noch nicht bereits Teil des Labels ist, z. B. "Inaktive Minuten/Tag")
    feature_units_de = {
        "age": "Jahre",
        "income": "Verhältnis zur Armutsgrenze",
        "kcal": "kcal/Tag",
        "protein": "g/Tag",
        "carbs": "g/Tag",
        "sugar": "g/Tag",
        "fiber": "g/Tag",
        "fat": "g/Tag",
        "sleep_avg": "Stunden/Tag"
    }

    def label_with_unit(key):
        unit = feature_units_de.get(key)
        return f"{feature_labels_de[key]} ({unit})" if unit else feature_labels_de[key]
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Übersicht", "🔍 Analyse", "🤖 ML-Modell", "📁 Daten upload"])

    with tab1:
        st.subheader("Gefilterte Daten")
        st.write(f"Zeige **{len(filtered_df)}** von {len(df)} Patienten, **{filtered_df.shape[1]}** von {df.shape[1]} Features")

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Patienten", len(filtered_df))
        col2.metric(f"Ø {label_with_unit('age')}", f"{filtered_df['age'].mean():.1f}")
        col3.metric("Ø inaktive Minuten/Tag", f"{filtered_df['sedentary_act_min'].mean():.0f}")

        # Daten anzeigen
        if st.checkbox("Zeige Rohdaten"):
            st.dataframe(filtered_df)

        if st.checkbox("Zeige statistische Kennzahlen"):
            st.write(filtered_df.describe())

    with tab2:
        # Visualisierung: BMI-Verteilung & Geschlechterverteilung
        st.subheader("Übersicht über Studienpopulation")

        col_pie, col_bar = st.columns(2)

        with col_pie:
            bmi_counts = filtered_df["bmi_cat"].value_counts().reindex(bmi_order, fill_value=0)

            fig_pie = px.pie(
                names=bmi_labels,
                values=bmi_counts.values,
                color=bmi_labels,
                color_discrete_map=dict(zip(bmi_labels, bmi_colors)),
                title="Verteilung der BMI-Kategorien"
            )
            fig_pie.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_bar:
            gender_order = [1.0, 2.0]
            gender_labels = ["Männer", "Frauen"]

            gender_counts = filtered_df["gender"].value_counts().reindex(gender_order, fill_value=0)

            fig_gender = px.bar(
                x=gender_labels,
                y=gender_counts.values,
                color=gender_labels,
                color_discrete_map={"Männer": "blue", "Frauen": "red"},
                labels={"x": "Geschlecht", "y": "Anzahl"},
                title="Geschlechterverteilung"
            )
            fig_gender.update_layout(showlegend=False)
            st.plotly_chart(fig_gender, use_container_width=True)

        # Visualisierung: Metrische Variablen
        st.markdown("---")
        st.subheader("Metrische Variablen")
        
        feature = st.selectbox( 
            "Feature wählen:", 
            ["age", "sleep_avg", "sedentary_act_min", "kcal", "protein", "carbs", "sugar", "fiber", "fat", "income"], 
            format_func=lambda x: label_with_unit(x)  
        )

        fig = px.histogram(
            filtered_df,
            x=feature,
            nbins=20,
            color_discrete_sequence=["skyblue"],
            labels={feature: label_with_unit(feature)},
            title=f"Verteilung: {label_with_unit(feature)}"
        )
        
        fig.update_traces(marker_line_color="black", marker_line_width=1)
        fig.update_layout(yaxis_title="Häufigkeit")
        st.plotly_chart(fig, use_container_width=True)

        # Visualisierung: Ordinale Variablen
        st.markdown("---")
        st.subheader("Ordinale Variablen")

        ordinal_config = {
            "education_lvl": {
                "order": [1, 2, 3, 4, 5],
                "labels": ["< 9. Klasse", "9.-11. Klasse", "High School/GED", "Irgendein College-Abschluss", "College-Absolvent"],
                "colors": None,
            },
            "alcohol": {
                "order": [0, 1, 2, 3, 4],
                "labels": ["nie", "selten", "moderat", "häufig", "(fast) täglich"],
                "colors": ["green", "yellowgreen", "yellow", "orange", "red"],
            },
            "smoke_status": {
                "order": [0, 1, 2],
                "labels": ["Nie", "Ex-Raucher", "Raucher"],
                "colors": ["green", "orange", "red"],
            },
            "activity_level": {
                "order": [0, 1, 2],
                "labels": ["inaktiv", "moderat aktiv", "sehr aktiv"],
                "colors": None,
            },
        }

        ordinal_feature = st.selectbox(
            "Ordinale Variable wählen:",
            list(ordinal_config.keys()),
            format_func=lambda x: feature_labels_de[x]
        )

        config = ordinal_config[ordinal_feature]
        ordinal_counts = filtered_df[ordinal_feature].value_counts().reindex(config["order"], fill_value=0)

        if config["colors"] is not None:
            color_map = dict(zip(config["labels"], config["colors"]))
        else:
            color_map = None

        fig2 = px.bar(
            x=config["labels"],
            y=ordinal_counts.values,
            color=config["labels"] if color_map else None,
            color_discrete_map=color_map,
            labels={"x": feature_labels_de[ordinal_feature], "y": "Häufigkeit"},
            title=f"Verteilung: {feature_labels_de[ordinal_feature]}"
        )
        fig2.update_traces(marker_line_color="black", marker_line_width=1)
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

        # Visualisierung: Feature vs. BMI-Kategorie
        st.markdown("---")
        st.subheader("Merkmale nach BMI-Kategorie")

        vs_bmi_feature = st.selectbox(
            "Merkmal wählen:",
            ["age", "kcal", "fiber", "sedentary_act_min", "sugar"],
            format_func=lambda x: label_with_unit(x),
            key="vs_bmi_select"
        )

        fig_box = px.box(
            filtered_df,
            x="bmi_cat",
            y=vs_bmi_feature,
            color="bmi_cat",
            color_discrete_map={0: "green", 1: "orange", 2: "red"},
            labels={"bmi_cat": "BMI-Kategorie", vs_bmi_feature: label_with_unit(vs_bmi_feature)},
            title=f"{label_with_unit(vs_bmi_feature)} nach BMI-Kategorie"
        )
        fig_box.update_layout(
            showlegend=False,
            xaxis=dict(tickmode="array", tickvals=[0, 1, 2], ticktext=bmi_labels)
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # Geschlecht nach BMI
        st.markdown("---")
        st.subheader("Geschlecht nach BMI-Kategorie")
        gender_bmi_df = filtered_df.copy()
        gender_bmi_df["gender_str"] = gender_bmi_df["gender"].map({1.0: "Männer", 2.0: "Frauen"})
        gender_bmi_df["bmi_str"] = gender_bmi_df["bmi_cat"].map(dict(enumerate(bmi_labels)))

        fig_gender_bmi = px.histogram(
            gender_bmi_df,
            x="gender_str",
            color="bmi_str",
            barmode="group",
            category_orders={"gender_str": ["Männer", "Frauen"], "bmi_str": bmi_labels},
            color_discrete_map=dict(zip(bmi_labels, bmi_colors)),
            labels={"gender_str": "Geschlecht", "count": "Anzahl", "bmi_str": "BMI-Kategorie"},
            title="BMI-Kategorie nach Geschlecht"
        )
        st.plotly_chart(fig_gender_bmi, use_container_width=True)

        # Korrelationsmatrix
        st.markdown("---")
        st.subheader("Korrelationen zwischen Merkmalen")

        corr_features = [
            "age", "education_lvl", "income", "kcal", "protein", "carbs",
            "sugar", "fiber", "fat", "alcohol", "sedentary_act_min",
            "smoke_status", "activity_level", "sleep_avg", "bmi_cat"
        ]

        corr_matrix = filtered_df[corr_features].corr().round(2)
        corr_matrix_display = corr_matrix.rename(index=feature_labels_de, columns=feature_labels_de)

        fig_corr = px.imshow(
            corr_matrix_display,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Korrelationsmatrix (Pearson)"
        )
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab3:
        st.header("🔮 BMI-Kategorie vorhersagen")
        st.write("Es handelt sich um ein Random Forest Modell. Leider konnte trotz Feature Engineering und Hyperparameter Tuning nur ein macro F1-Score von 44,2% erreicht werden. Damit ist das Modell zwar leicht besser als der Zufall (33,3%), aber hat eine schlechte Performance. Dies liegt vermutlich an der schwachen Korrelation der Features mit dem Target (s. Analyse).")

        st.markdown("---")
        st.subheader("📈 Feature Importance")
        st.write("Diese Grafik zeigt, welche Merkmale das Modell am stärksten für die Vorhersage nutzt.")

        model = load_model()
        medians = load_medians()
        selector = model.named_steps["selector"]
        classifier = model.named_steps["classifier"]

        selected_mask = selector.get_support()
        selected_features = np.array(COLUMN_ORDER)[selected_mask]
        importances = classifier.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": [feature_labels_de.get(f, f) for f in selected_features],
            "Wichtigkeit": importances
        }).sort_values("Wichtigkeit", ascending=True)

        fig_importance = px.bar(
            importance_df,
            x="Wichtigkeit",
            y="Feature",
            orientation="h",
            color_discrete_sequence=["#6FA8DC"],
            title="Feature Importance (Random Forest)"
        )
        fig_importance.update_layout(
            yaxis_title="",
            xaxis_title="Wichtigkeit",
            height=max(400, len(importance_df) * 30),
            yaxis=dict(tickmode="linear")
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        
        st.markdown("---") 
        st.subheader("Geben Sie die Patientendaten ein, um eine Vorhersage zu erhalten:")
        
        col1, col2, col3 = st.columns(3)
        
        # Input (Nutzung von min/max des Datensatzes als Eingrenzung
        with col1:
            age = st.number_input("Alter", 20, 80, 40)
            gender_input = st.selectbox("Geschlecht", [1.0, 2.0], format_func=lambda x: "Mann" if x == 1.0 else "Frau")
            education_lvl = st.selectbox("Bildungsniveau", [1, 2, 3, 4, 5], format_func=lambda x: {
                1: "< 9. Klasse", 2: "9.-11. Klasse", 3: "High School/GED",
                4: "Irgendein College-Abschluss", 5: "College-Absolvent"
            }[x])
            income = st.number_input("Einkommen (Verhältnis zur Armutsgrenze)", 0.0, 5.0, 4.0, step=0.01)
            alcohol = st.selectbox("Alkoholkonsum", [0, 1, 2, 3, 4], format_func=lambda x: {
                0: "nie", 1: "selten", 2: "moderat", 3: "häufig", 4: "(fast) täglich"
            }[x])
            
        with col2:
            kcal = st.number_input("Kalorien (kcal/Tag)", 249.0, 10446.0, 2000.0, step=1.0)
            protein = st.number_input("Protein (g/Tag)", 6.0, 460.0, 70.0, step=0.1)
            carbs = st.number_input("Kohlenhydrate (g/Tag)", 3.0, 1211.0, 200.0, step=0.1)
            sugar = st.number_input("Zucker (g/Tag)", 1.0, 510.0, 80.0, step=0.1)
            fiber = st.number_input("Ballaststoffe (g/Tag)", 0.2, 117.0, 20.0, step=0.1)
            fat = st.number_input("Fett (g/Tag)", 3.0, 506.0, 80.0, step=0.1)
        
        with col3:
            sedentary_act_min = st.number_input("Inaktive Minuten/Tag", 30.0, 1320.0, 360.0, step=1.0)
            smoke_status = st.selectbox("Rauchstatus", [0, 1, 2], format_func=lambda x: {
                0: "Nie", 1: "Ex-Raucher", 2: "Raucher"
            }[x])
            activity_level = st.selectbox("Aktivitätslevel", [0, 1, 2], format_func=lambda x: {
                0: "inaktiv", 1: "moderat aktiv", 2: "sehr aktiv"
            }[x])
            sleep_avg = st.number_input("Ø Schlafdauer (Std.)", 2.0, 13.7, 8.0, step=0.1)

        if st.button("Vorhersagen", type="primary"):
            base = {
                "gender": gender_input, "age": age, "education_lvl": education_lvl,
                "income": income, "kcal": kcal, "protein": protein, "carbs": carbs,
                "sugar": sugar, "fiber": fiber, "fat": fat, "alcohol": alcohol,
                "sedentary_act_min": sedentary_act_min, "smoke_status": smoke_status,
                "activity_level": activity_level, "sleep_avg": sleep_avg
            }
            
            # Feature Engineering nachbilden + Prediction
            input_df = engineer_features(pd.DataFrame([base]), medians)[COLUMN_ORDER]
                
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
                
            st.write("---")
            st.subheader("Ergebnis")
                
            if prediction == 0:
                st.success(f"Vorhergesagte Kategorie: **{bmi_labels[0]}** (Wahrscheinlichkeit: {probability[0]:.1%})")
            elif prediction == 1:
                st.warning(f"Vorhergesagte Kategorie: **{bmi_labels[1]}** (Wahrscheinlichkeit: {probability[1]:.1%})")
            else:
                st.error(f"Vorhergesagte Kategorie: **{bmi_labels[2]}** (Wahrscheinlichkeit: {probability[2]:.1%})")
            
            # Darstellung der Vorhersage
            col1, col2, col3 = st.columns(3)
            col1.metric(bmi_labels[0], f"{probability[0]:.1%}")
            col2.metric(bmi_labels[1], f"{probability[1]:.1%}")
            col3.metric(bmi_labels[2], f"{probability[2]:.1%}")
                
            fig_proba = px.bar(
                x=bmi_labels,
                y=probability,
                color=bmi_labels,
                color_discrete_map=dict(zip(bmi_labels, bmi_colors)),
                labels={"x": "BMI-Kategorie", "y": "Wahrscheinlichkeit"},
                title="Vorhersage-Wahrscheinlichkeiten"
            )
            fig_proba.update_traces(
                marker_line_color="black",
                marker_line_width=1,
                texttemplate="%{y:.1%}",
                textposition="outside"
            )
            fig_proba.update_layout(
                showlegend=False,
                yaxis_tickformat=".0%",
                yaxis_range=[0, 1]
            )
            st.plotly_chart(fig_proba, use_container_width=True)

    with tab4:
        st.header("📁 Mehrere Patienten hochladen (Batch-Vorhersage)")
        st.write("Laden Sie eine CSV-Datei mit mehreren Patienten hoch, um für alle gleichzeitig eine Vorhersage zu erhalten.")

        with st.expander("ℹ️ Anforderungen an die CSV-Datei anzeigen"):
            st.write("Die Datei muss folgende Spalten enthalten (Reihenfolge egal, zusätzliche Spalten werden ignoriert):")

            # Formatvorgabe für upload file
            column_info = pd.DataFrame([
                {"Spalte": "gender", "Typ": "float", "Werte/Bereich": "1.0 = Mann, 2.0 = Frau"},
                {"Spalte": "age", "Typ": "int", "Werte/Bereich": "20 – 80"},
                {"Spalte": "education_lvl", "Typ": "int", "Werte/Bereich": "1 (< 9. Klasse) bis 5 (College-Absolvent)"},
                {"Spalte": "income", "Typ": "float", "Werte/Bereich": "0.0 – 5.0 (Einkommen-Armutsgrenze-Verhältnis)"},
                {"Spalte": "kcal", "Typ": "float", "Werte/Bereich": "249 – 10446 (kcal/Tag)"},
                {"Spalte": "protein", "Typ": "float", "Werte/Bereich": "6 – 460 (g/Tag)"},
                {"Spalte": "carbs", "Typ": "float", "Werte/Bereich": "3 – 1211 (g/Tag)"},
                {"Spalte": "sugar", "Typ": "float", "Werte/Bereich": "1 – 510 (g/Tag)"},
                {"Spalte": "fiber", "Typ": "float", "Werte/Bereich": "0.2 – 117 (g/Tag)"},
                {"Spalte": "fat", "Typ": "float", "Werte/Bereich": "3 – 506 (g/Tag)"},
                {"Spalte": "alcohol", "Typ": "int", "Werte/Bereich": "0 (nie) bis 4 ((fast) täglich)"},
                {"Spalte": "sedentary_act_min", "Typ": "float", "Werte/Bereich": "30 – 1320 (Minuten/Tag)"},
                {"Spalte": "smoke_status", "Typ": "int", "Werte/Bereich": "0 = Nie, 1 = Ex-Raucher, 2 = Raucher"},
                {"Spalte": "activity_level", "Typ": "int", "Werte/Bereich": "0 = inaktiv, 1 = moderat, 2 = sehr aktiv"},
                {"Spalte": "sleep_avg", "Typ": "float", "Werte/Bereich": "2.0 – 13.7 (Stunden)"},
            ])
            st.dataframe(column_info, hide_index=True, use_container_width=True)

            st.write("Beispiel-Zeile:")
            st.code(
                "gender,age,education_lvl,income,kcal,protein,carbs,sugar,fiber,fat,alcohol,sedentary_act_min,smoke_status,activity_level,sleep_avg\n"
                "1.0,34,3,2.15,2100,78.5,240.0,85.0,14.5,82.0,1,300,0,1,7.5",
                language="csv"
            )

        uploaded_file = st.file_uploader("CSV-Datei auswählen", type="csv")

        if uploaded_file is not None:
            try:
                upload_df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Datei konnte nicht gelesen werden: {e}")
                upload_df = None

            if upload_df is not None:
                missing_cols = [col for col in REQUIRED_COLUMNS if col not in upload_df.columns]

                if missing_cols:
                    st.error(f"Folgende Spalten fehlen in der Datei: {', '.join(missing_cols)}")
                else:
                    st.success(f"Datei geladen: {uploaded_file.name} ({len(upload_df)} Patienten)")

                    if st.checkbox("Rohdaten anzeigen", key="batch_raw"):
                        st.dataframe(upload_df)

                    if st.button("Vorhersagen für alle Patienten"):
                        predict_df = upload_df[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")

                        if predict_df.isnull().any().any():
                            bad_rows = predict_df[predict_df.isnull().any(axis=1)].index.tolist()
                            st.error(
                                f"Ungültige oder fehlende Werte in Zeile(n): {bad_rows}. "
                                "Bitte Datei prüfen (nur numerische Werte erlaubt)."
                            )
                        else:
                            eng_df = engineer_features(predict_df, medians)
                            input_df = eng_df[COLUMN_ORDER]

                            predictions = model.predict(input_df)
                            probabilities = model.predict_proba(input_df)

                            result_df = upload_df.copy()
                            result_df["Vorhersage"] = [bmi_labels[p] for p in predictions]
                            result_df["Wahrscheinlichkeit"] = probabilities.max(axis=1)
                            result_df = result_df[["Vorhersage", "Wahrscheinlichkeit"] + upload_df.columns.tolist()]

                            st.dataframe(result_df)

                            csv_result = result_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "Ergebnisse als CSV herunterladen",
                                data=csv_result,
                                file_name="vorhersagen.csv",
                                mime="text/csv"
                            )
    # Footer
    st.markdown("---")
    st.write("Datenquelle: NHANES 08/2021 - 08/2023")

if __name__ == "__main__":
    main()