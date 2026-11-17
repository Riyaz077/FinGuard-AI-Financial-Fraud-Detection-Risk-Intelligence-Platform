from pathlib import Path
import sys, time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.data_generator import FEATURES, generate_transactions
from src.modeling import train_and_save, score_transactions

st.set_page_config(page_title="FinGuard AI | Risk Intelligence", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:linear-gradient(135deg,#071426 0%,#0b1d33 48%,#071426 100%); color:#e8f1fb;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif!important; color:#f3f8ff!important;}
[data-testid="stSidebar"] {background:#09182a; border-right:1px solid #183650;}
[data-testid="stMetric"] {background:linear-gradient(145deg,#102943,#0c2035); padding:16px 18px; border:1px solid #1c4564; border-radius:14px;}
[data-testid="stMetricLabel"] {color:#9bb5cc!important;}
[data-testid="stMetricValue"] {color:#f5fbff!important;}
div[data-testid="stTabs"] button {color:#a9c5db;}
div.stButton>button {background:#06b6d4;color:#041522;border:0;border-radius:9px;font-weight:700;}
div.stButton>button:hover {background:#67e8f9;color:#041522;}
div[data-testid="stDataFrame"] {border:1px solid #1c4564;border-radius:12px;}
.small-muted {color:#9bb5cc;font-size:13px;}
.hero {padding:22px 26px;border:1px solid #1b4665;border-radius:18px;background:radial-gradient(circle at 85% 15%,#16486a 0%,#102b45 36%,#0b1d33 78%);margin-bottom:18px;}
.hero-kicker {color:#67e8f9;letter-spacing:2px;font-size:11px;font-weight:700;}
.hero-title {font-family:'Space Grotesk';font-size:34px;font-weight:700;color:#f4f9ff;margin:4px 0;}
.hero-sub {color:#b7cadd;font-size:14px;}
.pill {display:inline-block;padding:5px 10px;border-radius:99px;background:#123c50;color:#67e8f9;font-size:11px;font-weight:700;margin-right:6px;}
hr {border-color:#1b3c56;}
</style>
""", unsafe_allow_html=True)

DATA_PATH = ROOT / "data" / "transactions.csv"
MODEL_PATH = ROOT / "models" / "fraud_classifier.joblib"
ANOM_PATH = ROOT / "models" / "anomaly_detector.joblib"

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = generate_transactions()
    DATA_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df

@st.cache_resource
def load_models():
    if not MODEL_PATH.exists() or not ANOM_PATH.exists():
        train_and_save(load_data(), str(ROOT / "models"))
    return joblib.load(MODEL_PATH), joblib.load(ANOM_PATH)

df = load_data()
model, anomaly_model = load_models()
metrics_path = ROOT / "models" / "model_metrics.csv"
metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()

with st.sidebar:
    st.markdown("## 🛡️ FinGuard AI")
    st.caption("FINANCIAL RISK INTELLIGENCE")
    page = st.radio("WORKSPACE", ["Executive Overview", "Live Transaction Monitor",
                                  "Fraud Investigation", "Model Intelligence", "Data Explorer"])
    st.divider()
    threshold = st.slider("Fraud alert threshold", 0.10, 0.95, 0.50, 0.05)
    st.caption("Adjust the probability threshold used by the decision engine.")
    st.divider()
    st.markdown("**System status**")
    st.markdown("🟢 Classifier operational")
    st.markdown("🟢 Anomaly detector operational")
    st.markdown("🟢 Data pipeline ready")
    st.caption("Demo environment • Synthetic data")

scored = score_transactions(df, model, anomaly_model, threshold)
fraud_count = int(scored["is_fraud"].sum())
fraud_rate = scored["is_fraud"].mean() * 100
flagged = int((scored["decision"] != "Approve").sum())
exposure = scored.loc[scored["is_fraud"] == 1, "amount"].sum()

def header(title, subtitle):
    st.markdown(f'<div class="hero"><div class="hero-kicker">FINGUARD AI / RISK OPERATIONS</div><div class="hero-title">{title}</div><div class="hero-sub">{subtitle}</div></div>', unsafe_allow_html=True)

def money(x):
    return f"₹{x:,.0f}"

if page == "Executive Overview":
    header("Financial Risk Command Center", "A unified view of transaction activity, fraud exposure and model-driven risk signals.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Transactions monitored", f"{len(scored):,}")
    c2.metric("Confirmed synthetic fraud", f"{fraud_count:,}", f"{fraud_rate:.2f}% of volume")
    c3.metric("Transactions requiring review", f"{flagged:,}")
    c4.metric("Fraud exposure (labelled)", money(exposure))
    st.markdown("### Transaction intelligence")
    left,right = st.columns([1.55,1])
    with left:
        daily = scored.assign(date=pd.to_datetime(scored.timestamp).dt.date).groupby("date").agg(
            transactions=("transaction_id","count"), flagged=("decision",lambda x:(x!="Approve").sum())
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily.date,y=daily.transactions,name="Transactions",mode="lines",
                                 line=dict(color="#22d3ee",width=2),fill="tozeroy"))
        fig.add_trace(go.Scatter(x=daily.date,y=daily.flagged,name="Flagged",mode="lines",
                                 line=dict(color="#fb923c",width=2)))
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                          height=330,margin=dict(l=10,r=10,t=25,b=10),legend=dict(orientation="h"))
        st.plotly_chart(fig,use_container_width=True)
    with right:
        counts = scored.decision.value_counts().rename_axis("Decision").reset_index(name="Count")
        fig = px.pie(counts,names="Decision",values="Count",hole=.66,
                     color="Decision",color_discrete_map={"Approve":"#22c55e","Investigate":"#fbbf24","Review / Block":"#fb7185"})
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",height=330,
                          margin=dict(l=5,r=5,t=25,b=5),legend=dict(orientation="h",y=-.1))
        st.plotly_chart(fig,use_container_width=True)
    st.markdown("### Latest high-risk signals")
    view = scored.sort_values("fraud_probability",ascending=False).head(8)
    st.dataframe(view[["transaction_id","timestamp","amount","risk_score","anomaly_flag","decision"]],
                 use_container_width=True,hide_index=True)
    st.caption("All figures are generated from a reproducible synthetic dataset. They are not real banking activity.")

elif page == "Live Transaction Monitor":
    header("Live Transaction Monitor", "Simulate an incoming transaction stream and score each event as it arrives.")
    a,b = st.columns([1,2])
    with a:
        batch_size = st.select_slider("Transactions per simulation", options=[5,10,20,30,50], value=10)
        speed = st.select_slider("Simulation speed", options=["Instant","Fast","Slow"], value="Fast")
        run = st.button("▶ Start transaction simulation",use_container_width=True)
        st.markdown('<div class="small-muted">Simulation replays randomly sampled records from the demo dataset.</div>',unsafe_allow_html=True)
    if run:
        sample = df.sample(batch_size, random_state=int(time.time()) % 10000).reset_index(drop=True)
        slot = st.empty()
        for i in range(1,len(sample)+1):
            current = score_transactions(sample.iloc[:i],model,anomaly_model,threshold).tail(1)
            row = current.iloc[0]
            color = "🔴" if row.decision == "Review / Block" else ("🟠" if row.decision == "Investigate" else "🟢")
            slot.markdown(f"**{color} Event {i}/{len(sample)}** — `{row.transaction_id}` · {money(row.amount)} · Risk **{row.risk_score:.1f}/100** · **{row.decision}**")
            if speed == "Slow": time.sleep(.65)
            elif speed == "Fast": time.sleep(.15)
        result = score_transactions(sample,model,anomaly_model,threshold)
        st.success(f"Simulation complete. {int((result.decision!='Approve').sum())} of {len(result)} events require attention.")
        st.dataframe(result[["transaction_id","amount","fraud_probability","risk_score","anomaly_flag","decision"]],
                     use_container_width=True,hide_index=True)
    st.markdown("### Manual transaction risk check")
    with st.form("manual_txn"):
        x1,x2,x3 = st.columns(3)
        amount = x1.number_input("Transaction amount (₹)",min_value=1.0,value=2500.0,step=100.0)
        hour = x2.slider("Hour of day",0,23,2)
        distance = x3.number_input("Distance from home (km)",0.0,2000.0,350.0)
        x4,x5,x6 = st.columns(3)
        velocity = x4.slider("Transactions in last hour",0,18,6)
        merchant = x5.slider("Merchant risk",0.0,1.0,.75)
        trust = x6.slider("Device trust",0.0,1.0,.2)
        submitted = st.form_submit_button("Analyse transaction")
    if submitted:
        manual = pd.DataFrame([{
            "amount":amount,"hour":hour,"day_of_week":2,"distance_from_home_km":distance,
            "transactions_last_1h":velocity,"avg_amount_24h":800.0,"merchant_risk":merchant,
            "device_trust":trust,"international":1,"card_present":0
        }])
        out = score_transactions(manual,model,anomaly_model,threshold).iloc[0]
        st.metric("Predicted fraud risk",f"{out.risk_score:.1f}/100")
        st.write(f"**Decision:** {out.decision}")
        st.progress(float(out.fraud_probability))
        st.caption("Model output is a decision-support signal, not a guarantee of fraud.")

elif page == "Fraud Investigation":
    header("Fraud Investigation Desk", "Prioritise suspicious transactions and inspect the signals contributing to risk.")
    high = scored[scored.decision != "Approve"].copy().sort_values("fraud_probability",ascending=False)
    st.metric("Open investigation queue",f"{len(high):,}")
    search = st.text_input("Search transaction ID")
    if search:
        high = high[high.transaction_id.str.contains(search,case=False,na=False)]
    st.dataframe(high[["transaction_id","timestamp","amount","risk_score","fraud_probability",
                       "anomaly_flag","international","device_trust","decision"]],
                 use_container_width=True,hide_index=True)
    if not high.empty:
        chosen = st.selectbox("Select a transaction to investigate",high.transaction_id.tolist())
        row = high[high.transaction_id == chosen].iloc[0]
        st.markdown("### Investigation summary")
        q1,q2,q3 = st.columns(3)
        q1.metric("Amount",money(row.amount))
        q2.metric("Risk score",f"{row.risk_score:.1f}/100")
        q3.metric("Anomaly signal","Detected" if row.anomaly_flag else "Not detected")
        reasons=[]
        if row.amount > 900: reasons.append("Unusually high transaction amount")
        if row.hour <= 4: reasons.append("Transaction occurred during a low-activity hour")
        if row.distance_from_home_km > 100: reasons.append("Transaction location is far from the usual location")
        if row.transactions_last_1h >= 5: reasons.append("High transaction velocity in the last hour")
        if row.device_trust < .35: reasons.append("Low device trust score")
        if row.merchant_risk > .65: reasons.append("Elevated merchant risk")
        if row.international: reasons.append("International transaction")
        st.markdown("**Potential risk indicators**")
        if reasons:
            for reason in reasons: st.write("•",reason)
        else: st.write("No simple rule-based indicators were triggered; review the model score and context.")
        st.download_button("Download investigation record",row.to_frame().T.to_csv(index=False),
                           file_name=f"{chosen}_investigation.csv",mime="text/csv")
    else:
        st.info("No transactions match the current filter.")

elif page == "Model Intelligence":
    header("Model Intelligence & Explainability", "Compare supervised classifiers and inspect the features associated with fraud risk.")
    if not metrics.empty:
        st.dataframe(metrics.style.format({c:"{:.3f}" for c in metrics.columns if c!="model"}),
                     use_container_width=True,hide_index=True)
        fig = px.bar(metrics.melt(id_vars="model",value_vars=["precision","recall","f1","roc_auc"],
                                  var_name="Metric",value_name="Score"),
                     x="Metric",y="Score",color="model",barmode="group",template="plotly_dark")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=350)
        st.plotly_chart(fig,use_container_width=True)
    try:
        forest = joblib.load(ROOT/"models"/"model_comparison.joblib")["Random Forest"]
        rf = forest.named_steps["model"]
        importance = pd.DataFrame({"Feature":FEATURES,"Importance":rf.feature_importances_}).sort_values("Importance")
        fig = px.bar(importance,x="Importance",y="Feature",orientation="h",template="plotly_dark",
                     title="Random Forest feature importance")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=420)
        st.plotly_chart(fig,use_container_width=True)
    except Exception as exc:
        st.warning(f"Feature importance unavailable: {exc}")
    st.info("ROC-AUC evaluates ranking quality across thresholds. Precision, recall and F1 depend on the selected operating threshold. Feature importance indicates model reliance, not causation.")

elif page == "Data Explorer":
    header("Data Explorer", "Explore the synthetic transaction population, feature distributions and labelled outcomes.")
    f1,f2,f3 = st.columns(3)
    min_amt,max_amt = float(df.amount.min()),float(df.amount.max())
    amount_range = f1.slider("Amount range (₹)",min_amt,max_amt,(min_amt,max_amt))
    fraud_filter = f2.selectbox("Label filter",["All","Fraud only","Non-fraud only"])
    hour_filter = f3.multiselect("Hour of day",list(range(24)),default=list(range(24)))
    filtered = df[df.amount.between(*amount_range) & df.hour.isin(hour_filter)]
    if fraud_filter == "Fraud only": filtered=filtered[filtered.is_fraud==1]
    if fraud_filter == "Non-fraud only": filtered=filtered[filtered.is_fraud==0]
    st.write(f"**{len(filtered):,}** transactions match the filters.")
    l,r=st.columns(2)
    with l:
        fig=px.histogram(filtered,x="amount",color=filtered.is_fraud.map({0:"Legitimate",1:"Fraud"}),
                         nbins=45,barmode="overlay",opacity=.78,template="plotly_dark",
                         labels={"color":"Synthetic label"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=330)
        st.plotly_chart(fig,use_container_width=True)
    with r:
        by_hour=filtered.groupby("hour",as_index=False).agg(transactions=("transaction_id","count"),fraud=("is_fraud","sum"))
        fig=px.bar(by_hour,x="hour",y=["transactions","fraud"],barmode="group",template="plotly_dark")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=330)
        st.plotly_chart(fig,use_container_width=True)
    st.dataframe(filtered.head(500),use_container_width=True,hide_index=True)
    st.download_button("Export filtered transactions",filtered.to_csv(index=False),
                       "finguard_filtered_transactions.csv","text/csv")

st.markdown("""
<style>
.finguard-footer { margin-top:34px; padding:34px 12px 18px; text-align:center; border-top:1px solid #20364b; border-bottom:1px solid #20364b; }
.finguard-footer-title { color:#35d9f4; font-family:'Space Grotesk',sans-serif; font-size:16px; font-weight:700; line-height:1.5; margin:18px 0 16px; }
.finguard-footer-label { color:#82a8cc; font-size:12px; margin:0 0 12px; }
.finguard-footer-name { color:#f5f8ff; font-family:'Space Grotesk',sans-serif; font-size:26px; font-weight:700; line-height:1.3; margin:0 0 10px; }
.finguard-footer-detail { color:#82a8cc; font-size:12px; line-height:1.9; margin:0; }
.finguard-footer-copyright { color:#82a8cc; font-size:11px; text-align:center; margin:18px 0 4px; }
</style>
<div class="finguard-footer">
  <div class="finguard-footer-title">FinGuard AI – Financial Fraud Detection &amp; Risk Intelligence Platform</div>
  <div class="finguard-footer-label">Designed &amp; Developed by</div>
  <div class="finguard-footer-name">Mohamed Riyaz S.</div>
  <p class="finguard-footer-detail">Integrated M.Tech CSE | Business Analytics</p>
  <p class="finguard-footer-detail">VIT Chennai</p>
</div>
<div class="finguard-footer-copyright">© 2026 Mohamed Riyaz S. | All Rights Reserved</div>
""", unsafe_allow_html=True)
