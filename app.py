import os
import time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="AI-Powered NIDS", layout="wide", initial_sidebar_state="expanded")

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv")
TEST_PATH = os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "unsw_nb15_random_forest.joblib")

def clean_dataset(filepath):
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.drop(columns=['id', 'attack_cat'], errors='ignore')
    
    for col in df.columns:
        if col != 'label' and df[col].dtype == 'object':
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().mean() > 0.95:
                df[col] = converted
    return df

# ----------------- SIDEBAR -----------------
st.sidebar.title("🛡️ SecureNet NIDS")
st.sidebar.caption("Information Security CLO 4 Platform")

menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Data Explorer", "Model Training", "Visualizations", "Threshold Tuning", "Predictions", "Live Flow Simulation"]
)

st.sidebar.markdown("---")
threshold = st.sidebar.slider(
    "Decision Threshold (Alarm Cut-off)",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="Higher threshold lowers false positive alerts; lower threshold maximizes attack recall."
)

data_found = os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH)
model_found = os.path.exists(MODEL_PATH)

st.sidebar.markdown("---")
st.sidebar.write(f"Dataset: {'🟢 Ready' if data_found else '🔴 Missing'}")
st.sidebar.write(f"Model:   {'🟢 Active' if model_found else '🟡 Untrained'}")

# ----------------- DASHBOARD -----------------
if menu == "Dashboard":
    st.title("🛡️ SecureNet SOC Intrusion Detection System")
    st.markdown("Machine learning network-flow classifier trained on the UNSW-NB15 dataset.")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pipeline Architecture", "Random Forest + Scaler")
    col2.metric("Target Output", "Binary (Normal / Attack)")
    col3.metric("Decision Threshold", f"{threshold:.2f}")
    col4.metric("Status", "Operational" if model_found else "Requires Training")
    
    st.markdown("---")
    st.subheader("SOC Analyst Briefing")
    st.info(
        "Standard perimeter controls fail against modern zero-day attacks. This engine evaluates "
        "flow timing, packet intervals, TTL ratios, and service protocols. Use the **Threshold Tuning** "
        "tab to directly balance alert fatigue against missed attacks."
    )

# ----------------- DATA EXPLORER -----------------
elif menu == "Data Explorer":
    st.title("📊 Dataset Inspection")
    if data_found:
        df_preview = pd.read_csv(TRAIN_PATH, nrows=5000)
        st.dataframe(df_preview.head(10), use_container_width=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Preview Records", f"{df_preview.shape[0]:,}")
        c2.metric("Telemetry Features", f"{df_preview.shape[1]}")
        c3.metric("Missing Values Spotted", f"{df_preview.isnull().sum().sum()}")
    else:
        st.error("Please place UNSW-NB15 CSV files into the `data/` directory.")

# ----------------- MODEL TRAINING -----------------
elif menu == "Model Training":
    st.title("⚙️ Model Training & Hyperparameters")
    
    if not data_found:
        st.error("Missing dataset inside `data/`.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            n_trees = st.slider("Number of Trees (n_estimators)", 20, 150, 80, 10)
            max_depth = st.slider("Maximum Depth", 5, 25, 15)
        with col2:
            train_split = st.slider("Training Split (%)", 50, 90, 75)
            sample_limit = st.select_slider(
                "Training Record Cap (Speed Boost)",
                options=[25000, 50000, 100000, "Full Dataset"],
                value=50000
            )
            
        if st.button("🚀 Train & Save Pipeline", use_container_width=True):
            with st.spinner("Fitting Random Forest with standard scaling and one-hot encoding..."):
                train_df = clean_dataset(TRAIN_PATH)
                if sample_limit != "Full Dataset" and len(train_df) > sample_limit:
                    train_df = train_df.sample(sample_limit, random_state=42)
                    
                X = train_df.drop(columns=['label'])
                y = train_df['label'].astype(int)
                
                num_cols = X.select_dtypes(include=np.number).columns.tolist()
                cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
                
                preprocessor = ColumnTransformer([
                    ("num", Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())
                    ]), num_cols),
                    ("cat", Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                    ]), cat_cols)
                ])
                
                pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("classifier", RandomForestClassifier(
                        n_estimators=n_trees,
                        max_depth=max_depth,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1
                    ))
                ])
                
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X, y, train_size=train_split / 100.0, random_state=42, stratify=y
                )
                
                pipeline.fit(X_tr, y_tr)
                joblib.dump(pipeline, MODEL_PATH)
                
                y_probs = pipeline.predict_proba(X_val)[:, 1]
                y_pred = (y_probs >= threshold).astype(int)
                
                st.success("Pipeline trained and exported to `models/`!")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Validation Accuracy", f"{accuracy_score(y_val, y_pred):.2%}")
                m2.metric("Precision", f"{precision_score(y_val, y_pred, zero_division=0):.2%}")
                m3.metric("Attack Recall", f"{recall_score(y_val, y_pred, zero_division=0):.2%}")
                m4.metric("F1-Score", f"{f1_score(y_val, y_pred, zero_division=0):.2%}")
                st.rerun()

# ----------------- VISUALIZATIONS -----------------
elif menu == "Visualizations":
    st.title("📈 Performance Metrics & Telemetry")
    if not model_found or not data_found:
        st.warning("Train the model first or verify dataset files.")
    else:
        pipeline = joblib.load(MODEL_PATH)
        test_df = clean_dataset(TEST_PATH)
        eval_sample = test_df.sample(min(10000, len(test_df)), random_state=42)
        X_test = eval_sample.drop(columns=['label'])
        y_test = eval_sample['label'].astype(int)
        
        y_probs = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_probs >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Confusion Matrix (Threshold: {threshold:.2f})**")
            fig, ax = plt.subplots(figsize=(5, 3.8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(fig)
            
        with c2:
            st.markdown("**ROC / AUC Curve**")
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(5, 3.8))
            ax.plot(fpr, tpr, color='darkorange', label=f'AUC = {roc_auc:.4f}')
            ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")
            st.pyplot(fig)

# ----------------- THRESHOLD TUNING -----------------
elif menu == "Threshold Tuning":
    st.title("⚖️ SOC Alert Threshold Optimization")
    st.markdown("Fine-tune decision probability cut-offs to minimize false positive noise while preserving high recall.")
    
    if not model_found or not data_found:
        st.warning("Train the model and check datasets first.")
    else:
        pipeline = joblib.load(MODEL_PATH)
        test_df = clean_dataset(TEST_PATH)
        eval_sample = test_df.sample(min(10000, len(test_df)), random_state=42)
        X_test = eval_sample.drop(columns=['label'])
        y_test = eval_sample['label'].astype(int)
        y_probs = pipeline.predict_proba(X_test)[:, 1]
        
        threshold_range = np.linspace(0.1, 0.9, 17)
        prec_list, rec_list, fpr_list = [], [], []
        
        for t in threshold_range:
            p = (y_probs >= t).astype(int)
            prec_list.append(precision_score(y_test, p, zero_division=0))
            rec_list.append(recall_score(y_test, p, zero_division=0))
            cm = confusion_matrix(y_test, p)
            tn, fp, fn, tp = cm.ravel()
            fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
            
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.plot(threshold_range, rec_list, label="Attack Recall (Coverage)", marker='o')
        ax.plot(threshold_range, prec_list, label="Precision", marker='s')
        ax.plot(threshold_range, fpr_list, label="False Positive Rate (Noise)", marker='^', color='red')
        ax.axvline(threshold, color='black', linestyle='--', label=f'Current Selected Cut-off ({threshold:.2f})')
        ax.set_xlabel("Probability Threshold")
        ax.set_ylabel("Score")
        ax.legend()
        st.pyplot(fig)
        
        cur_pred = (y_probs >= threshold).astype(int)
        cm_cur = confusion_matrix(y_test, cur_pred)
        tn, fp, fn, tp = cm_cur.ravel()
        cur_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        cur_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Current False Positive Rate", f"{cur_fpr:.2%}")
        k2.metric("Current Attack Recall", f"{recall_score(y_test, cur_pred):.2%}")
        k3.metric("Missed Attacks (FNR)", f"{cur_fnr:.2%}")

# ----------------- PREDICTIONS -----------------
elif menu == "Predictions":
    st.title("🎯 Network Telemetry Prediction")
    if not model_found:
        st.error("Train the model first.")
    else:
        pipeline = joblib.load(MODEL_PATH)
        tab1, tab2 = st.tabs(["Manual Single Packet Entry", "Batch CSV Upload"])
        
        with tab1:
            col1, col2, col3 = st.columns(3)
            dur = col1.number_input("Duration (dur)", min_value=0.0, max_value=100.0, value=0.00105, format="%.5f")
            sbytes = col2.number_input("Source Bytes (sbytes)", min_value=0, value=146)
            dbytes = col3.number_input("Destination Bytes (dbytes)", min_value=0, value=178)
            
            col4, col5, col6 = st.columns(3)
            proto = col4.selectbox("Protocol (proto)", ["udp", "tcp", "unas", "ospf", "sctp"])
            state = col5.selectbox("State (state)", ["CON", "INT", "FIN", "REQ", "RST"])
            service = col6.selectbox("Service (service)", ["dns", "http", "ftp", "smtp", "ssh", "-"])
            
            if st.button("Run Threat Scan", use_container_width=True):
                prep = pipeline.named_steps["preprocessor"]
                num_cols = prep.transformers_[0][2]
                cat_cols = prep.transformers_[1][2]
                
                input_dict = {col: [0.0] for col in num_cols}
                for col in cat_cols:
                    input_dict[col] = ["-"]
                    
                sample_df = pd.DataFrame(input_dict)
                sample_df['dur'] = dur
                sample_df['sbytes'] = sbytes
                sample_df['dbytes'] = dbytes
                sample_df['proto'] = proto
                sample_df['state'] = state
                sample_df['service'] = service
                
                prob = pipeline.predict_proba(sample_df)[0][1]
                is_attack = prob >= threshold
                
                if is_attack:
                    st.error(f"🚨 **MALICIOUS TRAFFIC DETECTED** (Probability: {prob*100:.2f}%)")
                else:
                    st.success(f"✅ **NORMAL NETWORK FLOW** (Attack Probability: {prob*100:.2f}%)")
                    
        with tab2:
            uploaded_file = st.file_uploader("Upload flow records (.csv)", type=["csv"])
            if uploaded_file is not None:
                df_batch = pd.read_csv(uploaded_file)
                st.dataframe(df_batch.head(5), use_container_width=True)
                
                if st.button("Execute Batch Analysis"):
                    prep = pipeline.named_steps["preprocessor"]
                    num_cols = prep.transformers_[0][2]
                    cat_cols = prep.transformers_[1][2]
                    
                    df_aligned = pd.DataFrame(index=np.arange(len(df_batch)))
                    for col in num_cols:
                        df_aligned[col] = pd.to_numeric(df_batch[col], errors='coerce').fillna(0) if col in df_batch.columns else 0.0
                    for col in cat_cols:
                        df_aligned[col] = df_batch[col].astype(str).fillna('-') if col in df_batch.columns else '-'
                        
                    batch_probs = pipeline.predict_proba(df_aligned)[:, 1]
                    df_batch['Malicious_Probability'] = batch_probs
                    df_batch['SOC_Verdict'] = np.where(batch_probs >= threshold, '🚨 ATTACK', '✅ NORMAL')
                    
                    st.dataframe(df_batch[['SOC_Verdict', 'Malicious_Probability'] + [c for c in df_batch.columns if c not in ['SOC_Verdict', 'Malicious_Probability']]].head(50), use_container_width=True)

# ----------------- LIVE FLOW SIMULATION -----------------
elif menu == "Live Flow Simulation":
    st.title("📡 Real-Time Network Packet Simulator")
    st.caption("Sequential playback of unseen test flows against the trained classifier.")
    
    if not model_found or not data_found:
        st.warning("Ensure model is trained and data files exist.")
    else:
        pipeline = joblib.load(MODEL_PATH)
        test_df = clean_dataset(TEST_PATH)
        
        delay = st.slider("Simulated Packet Interval (seconds)", 0.05, 1.0, 0.2, 0.05)
        num_packets = st.slider("Stream Packet Count", 10, 100, 30)
        
        if st.button("Start Real-Time Stream", use_container_width=True):
            stream_sample = test_df.sample(num_packets, random_state=int(time.time()))
            ground_truth = stream_sample['label'].astype(int).values
            stream_features = stream_sample.drop(columns=['label'])
            
            status_box = st.empty()
            table_box = st.empty()
            stream_history = []
            
            for i in range(len(stream_features)):
                row = stream_features.iloc[[i]]
                prob = pipeline.predict_proba(row)[0][1]
                verdict = "🚨 ATTACK" if prob >= threshold else "✅ NORMAL"
                actual = "Attack" if ground_truth[i] == 1 else "Normal"
                
                stream_history.insert(0, {
                    "Packet #": i + 1,
                    "Verdict": verdict,
                    "Attack Probability": f"{prob * 100:.2f}%",
                    "Actual Flow": actual,
                    "Protocol": row['proto'].values[0] if 'proto' in row else 'N/A',
                    "Service": row['service'].values[0] if 'service' in row else 'N/A'
                })
                
                status_box.info(f"Ingesting packet #{i+1} of {num_packets}...")
                table_box.dataframe(pd.DataFrame(stream_history), use_container_width=True)
                time.sleep(delay)
                
            status_box.success("Stream simulation complete!")