import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import io

st.set_page_config(
    page_title="CreditGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, .stApp { background: #05070f !important; color: #e0e8f0; font-family: 'Syne', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1400px; margin: auto; position: relative; z-index: 1; }

/* Globe sits behind everything */
#globe-container {
    position: fixed;
    top: 50%;
    right: -180px;
    transform: translateY(-50%);
    width: 700px;
    height: 700px;
    z-index: 0;
    opacity: 0.18;
    pointer-events: none;
}
#globe-container canvas { border-radius: 50%; }

.hero-wrap { text-align: center; padding: 50px 20px 30px; }
.hero-badge { display:inline-block; font-family:'JetBrains Mono',monospace; font-size:0.7em; letter-spacing:6px; color:#3d7eff; border:1px solid #3d7eff55; border-radius:100px; padding:5px 18px; margin-bottom:20px; }
.hero-title { font-family:'Syne',sans-serif; font-size:clamp(2.8em,6vw,4.5em); font-weight:800; background:linear-gradient(135deg,#ffffff 30%,#3d7eff 70%,#00d4ff 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:12px; }
.hero-sub { color:#8899bb; font-size:1em; max-width:500px; margin:0 auto 30px; line-height:1.6; }

.stat-card { background:#080e1e; border:1px solid #1a2a4a; border-radius:16px; padding:22px 20px; position:relative; overflow:hidden; }
.stat-card::after { content:''; position:absolute; top:0; left:0; right:0; height:3px; background:var(--c); }
.stat-label { font-family:'JetBrains Mono',monospace; font-size:0.68em; color:#5577aa; letter-spacing:3px; text-transform:uppercase; margin-bottom:8px; }
.stat-val { font-size:2em; font-weight:800; color:var(--c); line-height:1; }
.stat-sub { font-size:0.75em; color:#445566; margin-top:4px; }

.alert-fraud { background:linear-gradient(135deg,#1a0505,#2d0808); border:1px solid #ff3c3c55; border-left:4px solid #ff3c3c; border-radius:16px; padding:20px 24px; margin:12px 0; }
.alert-safe  { background:linear-gradient(135deg,#020f08,#041a0d); border:1px solid #00cc6655; border-left:4px solid #00cc66; border-radius:16px; padding:20px 24px; margin:12px 0; }
.alert-title { font-size:1.2em; font-weight:800; margin-bottom:4px; }
.alert-fraud .alert-title { color:#ff5555; }
.alert-safe  .alert-title { color:#00cc66; }
.alert-desc  { color:#8899bb; font-size:0.88em; }

.section-hdr { font-family:'JetBrains Mono',monospace; font-size:0.72em; letter-spacing:5px; color:#3d7eff; text-transform:uppercase; margin:2rem 0 1rem; display:flex; align-items:center; gap:12px; }
.section-hdr::after { content:''; flex:1; height:1px; background:#1a2a4a; }

.demo-btn-row { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:10px; }

.stButton > button { background:linear-gradient(135deg,#1a3aff,#0066ff) !important; color:white !important; border:none !important; border-radius:10px !important; padding:10px 24px !important; font-family:'Syne',sans-serif !important; font-weight:700 !important; cursor:pointer !important; transition:all 0.2s !important; }
.stButton > button:hover { opacity:0.85 !important; }

.perf-box { background:#080e1e; border:1px solid #1a2a4a; border-radius:14px; padding:18px 22px; margin:8px 0; }
.perf-label { font-family:'JetBrains Mono',monospace; font-size:0.7em; color:#5577aa; letter-spacing:3px; margin-bottom:6px; }
.perf-val { font-size:1.6em; font-weight:800; }
</style>
""", unsafe_allow_html=True)


#  MODEL (cached so it trains only once)
@st.cache_resource(show_spinner="🤖 Training fraud detection model...")
def get_model():
    np.random.seed(42)
    n_legit, n_fraud = 8000, 400
    legit = np.random.randn(n_legit, 28) * 0.6
    fraud = np.random.randn(n_fraud, 28) * 1.8
    fraud[:, [0,1,2,3]] += np.random.choice([-3,3], size=(n_fraud,4))
    X = np.vstack([
        np.column_stack([np.random.uniform(0,172800,n_legit), legit, np.abs(np.random.lognormal(4,1.2,n_legit))]),
        np.column_stack([np.random.uniform(0,172800,n_fraud), fraud, np.abs(np.random.lognormal(6,1.0,n_fraud))])
    ])
    y = np.array([0]*n_legit + [1]*n_fraud)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.2, stratify=y, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X_s)
    # model performance on test set
    y_pred = rf.predict(X_te)
    perf = {
        'accuracy':  round(accuracy_score(y_te, y_pred)*100, 1),
        'precision': round(precision_score(y_te, y_pred, zero_division=0)*100, 1),
        'recall':    round(recall_score(y_te, y_pred, zero_division=0)*100, 1),
        'f1':        round(f1_score(y_te, y_pred, zero_division=0)*100, 1),
    }
    return rf, iso, scaler, perf


def predict_transactions(df):
    rf, iso, scaler, _ = get_model()
    v_cols   = [c for c in df.columns if c.upper().startswith('V')][:28]
    amt_col  = next((c for c in df.columns if 'amount' in c.lower()), None)
    time_col = next((c for c in df.columns if 'time'   in c.lower()), None)
    rows = []
    for _, r in df.iterrows():
        vs = [float(r[c]) if c in df.columns else 0.0 for c in v_cols]
        vs += [0.0]*(28-len(vs))
        rows.append([float(r[time_col]) if time_col else 0.0] + vs +
                    [float(r[amt_col])  if amt_col  else 0.0])
    X   = np.array(rows, dtype=np.float32)
    X_s = scaler.transform(X)
    rf_prob    = rf.predict_proba(X_s)[:,1]
    iso_raw    = iso.decision_function(X_s)
    iso_norm   = 1 - (iso_raw - iso_raw.min()) / ((iso_raw.max()-iso_raw.min()) + 1e-9)
    score      = 0.70*rf_prob + 0.30*iso_norm
    pred       = (score >= 0.45).astype(int)
    return pred, score


#  DEMO DATA GENERATORS 
def make_demo(seed, n_legit, n_fraud, label):
    np.random.seed(seed)
    rows = []
    for _ in range(n_legit):
        v = np.random.randn(28)*0.5
        rows.append({'Time': np.random.randint(0,172800),
                     **{f'V{j+1}': round(v[j],4) for j in range(28)},
                     'Amount': round(abs(np.random.lognormal(4,1.2)),2), 'Class':0})
    for _ in range(n_fraud):
        v = np.random.randn(28)*1.8
        v[[0,1,2,3]] += np.random.choice([-3,3],4)
        rows.append({'Time': np.random.randint(0,172800),
                     **{f'V{j+1}': round(v[j],4) for j in range(28)},
                     'Amount': round(abs(np.random.lognormal(6,1)),2), 'Class':1})
    return pd.DataFrame(rows).sample(frac=1,random_state=seed).reset_index(drop=True)

DEMOS = {
    "🟢 Low Fraud (5%)":   (make_demo(10,  95,  5,  "low"),   "95 legit, 5 fraud — typical day"),
    "🟡 Medium Fraud (20%)":(make_demo(20,  80, 20, "med"),   "80 legit, 20 fraud — suspicious activity"),
    "🔴 High Fraud (40%)":  (make_demo(30,  60, 40, "high"),  "60 legit, 40 fraud — card compromised!"),
}


# HERO 
st.markdown("""
<div class="hero-wrap">
  <div class="hero-badge">🛡️ AI-Powered Protection</div>
  <div class="hero-title">CreditGuard AI</div>
  <div class="hero-sub">Upload your transaction data — our AI instantly detects fraud and flags suspicious charges.</div>
</div>
""", unsafe_allow_html=True)

#  ROTATING GLOBE ( 
st.markdown("""
<canvas id="cg-globe"></canvas>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function boot() {
  if (typeof THREE === 'undefined') { setTimeout(boot, 150); return; }
  var cv = document.getElementById('cg-globe');
  if (!cv) return;
  var W = 700, H = 700;
  var renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: true, alpha: true });
  renderer.setSize(W, H);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);
  var scene  = new THREE.Scene();
  var camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
  camera.position.z = 2.4;

  // Globe body
  var globe = new THREE.Mesh(
    new THREE.SphereGeometry(1, 64, 64),
    new THREE.MeshPhongMaterial({ color:0x071428, emissive:0x0b1e40, specular:0x1a56db, shininess:80, transparent:true, opacity:0.93 })
  );
  scene.add(globe);

  // Grid wireframe
  var grid = new THREE.Mesh(
    new THREE.SphereGeometry(1.003, 26, 26),
    new THREE.MeshBasicMaterial({ color:0x1a56db, wireframe:true, transparent:true, opacity:0.22 })
  );
  scene.add(grid);

  // Atmosphere glow
  scene.add(new THREE.Mesh(
    new THREE.SphereGeometry(1.09, 32, 32),
    new THREE.MeshBasicMaterial({ color:0x1a56db, transparent:true, opacity:0.05, side:THREE.BackSide })
  ));

  // Dots
  var N=350, pos=[];
  for(var i=0;i<N;i++){
    var phi=Math.acos(-1+(2*i)/N), theta=Math.sqrt(N*Math.PI)*phi, r=1.013;
    pos.push(r*Math.sin(phi)*Math.cos(theta), r*Math.cos(phi), r*Math.sin(phi)*Math.sin(theta));
  }
  var dg=new THREE.BufferGeometry();
  dg.setAttribute('position', new THREE.Float32BufferAttribute(pos,3));
  var dots=new THREE.Points(dg, new THREE.PointsMaterial({color:0x00d4ff, size:0.020, transparent:true, opacity:0.85}));
  scene.add(dots);

  // Arcs
  var ac=[0x00d4ff,0x1a56db,0xff3c3c,0x00ffaa];
  for(var i=0;i<22;i++){
    var phi1=Math.random()*Math.PI, t1=Math.random()*Math.PI*2;
    var phi2=Math.random()*Math.PI, t2=Math.random()*Math.PI*2;
    var p1=new THREE.Vector3(Math.sin(phi1)*Math.cos(t1),Math.cos(phi1),Math.sin(phi1)*Math.sin(t1));
    var p2=new THREE.Vector3(Math.sin(phi2)*Math.cos(t2),Math.cos(phi2),Math.sin(phi2)*Math.sin(t2));
    var mid=p1.clone().add(p2).multiplyScalar(0.5).normalize().multiplyScalar(1.4);
    var pts=new THREE.QuadraticBezierCurve3(p1,mid,p2).getPoints(50);
    scene.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({color:ac[i%4], transparent:true, opacity:0.4})
    ));
  }

  // Ring
  var ring=new THREE.Mesh(
    new THREE.RingGeometry(1.12,1.17,100),
    new THREE.MeshBasicMaterial({color:0x1a56db, side:THREE.DoubleSide, transparent:true, opacity:0.13})
  );
  ring.rotation.x=1.1; scene.add(ring);

  // Lights
  scene.add(new THREE.AmbientLight(0x112244,1.5));
  var dl=new THREE.DirectionalLight(0x1a56db,2.5); dl.position.set(4,3,5); scene.add(dl);
  var pl=new THREE.PointLight(0x00d4ff,2.0,10); pl.position.set(-2,2,2); scene.add(pl);

  // Animate
  var t=0;
  (function loop(){
    requestAnimationFrame(loop); t+=0.0035;
    globe.rotation.y=grid.rotation.y=dots.rotation.y=t;
    globe.rotation.x=grid.rotation.x=dots.rotation.x=Math.sin(t*0.35)*0.07;
    ring.rotation.z=t*0.25;
    renderer.render(scene,camera);
  })();
})();
</script>
<style>
#cg-globe {
  position: fixed !important;
  top: 50% !important;
  right: -160px !important;
  transform: translateY(-50%) !important;
  width: 700px !important;
  height: 700px !important;
  opacity: 0.19 !important;
  pointer-events: none !important;
  z-index: 0 !important;
}
</style>
""", unsafe_allow_html=True)



# UPLOAD + DEMO BUTTONS
col_up, col_btns = st.columns([2, 2])

with col_up:
    uploaded = st.file_uploader("Upload Transactions CSV", type=["csv"],
                                 label_visibility="collapsed")

with col_btns:
    st.markdown("**⚡ Quick Demo Datasets**")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("🟢 Low\nFraud"):
            st.session_state['demo_key'] = "🟢 Low Fraud (5%)"
            st.session_state['use_demo'] = True
            st.rerun()
    with b2:
        if st.button("🟡 Medium\nFraud"):
            st.session_state['demo_key'] = "🟡 Medium Fraud (20%)"
            st.session_state['use_demo'] = True
            st.rerun()
    with b3:
        if st.button("🔴 High\nFraud"):
            st.session_state['demo_key'] = "🔴 High Fraud (40%)"
            st.session_state['use_demo'] = True
            st.rerun()

#  LOAD DATA
df_raw = None
demo_label = ""

if uploaded:
    df_raw = pd.read_csv(uploaded)
    st.session_state['use_demo'] = False
elif st.session_state.get('use_demo') and st.session_state.get('demo_key'):
    key = st.session_state['demo_key']
    df_raw, demo_label = DEMOS[key][0], DEMOS[key][1]

# LANDING PAGE
if df_raw is None:
    st.markdown('<div class="section-hdr">How it works</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    for col,(num,title,desc) in zip([c1,c2,c3],[
        ("01","Upload CSV","Drop your bank export or click a demo button above."),
        ("02","AI Scans","Random Forest + Isolation Forest ensemble scores every transaction."),
        ("03","See Results","Fraud flagged instantly with risk score, charts and report."),
    ]):
        with col:
            st.markdown(f"""<div class="stat-card" style="--c:#3d7eff">
                <div class="stat-label">Step {num}</div>
                <div style="font-size:1.1em;font-weight:800;color:#c0d4ff;margin-bottom:8px">{title}</div>
                <div style="color:#5577aa;font-size:0.85em;line-height:1.5">{desc}</div>
            </div>""", unsafe_allow_html=True)

   
    _, _, _, perf = get_model()
    st.markdown('<div class="section-hdr">Model Performance (on training test split)</div>', unsafe_allow_html=True)
    p1,p2,p3,p4 = st.columns(4)
    for col, label, val, color in zip([p1,p2,p3,p4],
        ["Accuracy","Precision","Recall","F1-Score"],
        [f"{perf['accuracy']}%",f"{perf['precision']}%",f"{perf['recall']}%",f"{perf['f1']}%"],
        ["#3d7eff","#ffd700","#00cc66","#ff9966"]):
        with col:
            st.markdown(f"""<div class="perf-box">
                <div class="perf-label">{label}</div>
                <div class="perf-val" style="color:{color}">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Expected CSV Format</div>', unsafe_allow_html=True)
    st.code("Time, V1, V2, ..., V28, Amount, Class(optional)", language="text")
    st.caption("Compatible with the Kaggle Credit Card Fraud Detection dataset.")
    st.stop()


#  RUN DETECTION 
with st.spinner("🔍 Scanning transactions for fraud..."):
    fraud_pred, fraud_score = predict_transactions(df_raw)

df_res = df_raw.copy()
df_res['_pred']  = fraud_pred
df_res['_score'] = fraud_score

n_total    = len(df_res)
n_fraud    = int(fraud_pred.sum())
n_safe     = n_total - n_fraud
fraud_rate = n_fraud / n_total * 100
amt_col    = next((c for c in df_raw.columns if 'amount' in c.lower()), None)
amt_risk   = df_res[df_res['_pred']==1][amt_col].sum() if amt_col else 0

#  RESET BUTTON
if st.button("🔄 Reset / Try Another Dataset"):
    st.session_state['use_demo'] = False
    st.session_state['demo_key'] = None
    st.rerun()

#  DEMO LABEL
if demo_label:
    st.info(f"📂 Demo dataset loaded: **{st.session_state.get('demo_key','')}** — {demo_label}")

#  SUMMARY STATS 
st.markdown('<div class="section-hdr">Protection Summary</div>', unsafe_allow_html=True)
c1,c2,c3,c4 = st.columns(4)
for col,label,val,sub,color in zip([c1,c2,c3,c4],
    ["Transactions Scanned","Fraud Detected","Safe Transactions","Amount at Risk"],
    [f"{n_total:,}", f"{n_fraud:,}", f"{n_safe:,}", f"₹{amt_risk:,.0f}" if amt_col else "N/A"],
    ["total uploaded","flagged by AI","cleared","needs action"],
    ["#3d7eff","#ff3c3c","#00cc66","#ffaa00"]):
    with col:
        st.markdown(f"""<div class="stat-card" style="--c:{color}">
            <div class="stat-label">{label}</div>
            <div class="stat-val">{val}</div>
            <div class="stat-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

#  ALERT 
if n_fraud == 0:
    st.markdown("""<div class="alert-safe">
        <div class="alert-title">✅ All Clear — No Fraud Detected</div>
        <div class="alert-desc">All transactions passed our AI security scan. Your card activity looks completely normal.</div>
    </div>""", unsafe_allow_html=True)
elif fraud_rate > 25:
    st.markdown(f"""<div class="alert-fraud">
        <div class="alert-title">🚨 HIGH RISK — {n_fraud} Fraudulent Transactions Detected! ({fraud_rate:.1f}%)</div>
        <div class="alert-desc">Your card may be compromised. Block it immediately and contact your bank.</div>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""<div class="alert-fraud">
        <div class="alert-title">⚠️ {n_fraud} Suspicious Transaction{'s' if n_fraud>1 else ''} Flagged ({fraud_rate:.1f}%)</div>
        <div class="alert-desc">Please review the flagged transactions below and verify with your bank.</div>
    </div>""", unsafe_allow_html=True)

# CHARTS 
st.markdown('<div class="section-hdr">Risk Analysis</div>', unsafe_allow_html=True)
cl, cr = st.columns([1,2])

with cl:
    fig_d = go.Figure(go.Pie(
        labels=['Safe','Fraud'], values=[n_safe, n_fraud], hole=0.70,
        marker_colors=['#00cc66','#ff3c3c'], textinfo='none'))
    fig_d.update_layout(paper_bgcolor='#080e1e', plot_bgcolor='#080e1e',
        font_color='white', height=260, margin=dict(t=10,b=10,l=10,r=10),
        showlegend=True, legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0)'),
        annotations=[dict(text=f"<b>{100-fraud_rate:.0f}%</b><br>Safe",
            x=0.5,y=0.5,font_size=17,font_color='#00cc66',showarrow=False)])
    st.plotly_chart(fig_d, use_container_width=True)

with cr:
    fig_h = go.Figure()
    s_scores = fraud_score[fraud_pred==0]
    f_scores = fraud_score[fraud_pred==1]
    if len(s_scores): fig_h.add_trace(go.Histogram(x=s_scores, name='Safe',  nbinsx=20, marker_color='#00cc66', opacity=0.75))
    if len(f_scores): fig_h.add_trace(go.Histogram(x=f_scores, name='Fraud', nbinsx=20, marker_color='#ff3c3c', opacity=0.75))
    fig_h.add_vline(x=0.45, line_dash='dash', line_color='#ffaa00',
                    annotation_text='Threshold 45%', annotation_font_color='#ffaa00')
    fig_h.update_layout(barmode='overlay', paper_bgcolor='#080e1e', plot_bgcolor='#080e1e',
        font_color='white', height=260, margin=dict(t=10,b=40,l=40,r=10),
        xaxis_title='Fraud Risk Score', yaxis_title='Count',
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(gridcolor='#1a2a4a'), yaxis=dict(gridcolor='#1a2a4a'))
    st.plotly_chart(fig_h, use_container_width=True)

# MODEL PERFORMANCE
_, _, _, perf = get_model()
st.markdown('<div class="section-hdr">Model Performance Metrics</div>', unsafe_allow_html=True)
p1,p2,p3,p4 = st.columns(4)
for col, label, val, color in zip([p1,p2,p3,p4],
    ["Accuracy","Precision","Recall","F1-Score"],
    [f"{perf['accuracy']}%",f"{perf['precision']}%",f"{perf['recall']}%",f"{perf['f1']}%"],
    ["#3d7eff","#ffd700","#00cc66","#ff9966"]):
    with col:
        st.markdown(f"""<div class="perf-box">
            <div class="perf-label">{label}</div>
            <div class="perf-val" style="color:{color}">{val}</div>
        </div>""", unsafe_allow_html=True)

#  FLAGGED TRANSACTIONS 
if n_fraud > 0:
    st.markdown('<div class="section-hdr">🚨 Flagged Fraud Transactions</div>', unsafe_allow_html=True)
    fdf = df_res[df_res['_pred']==1].copy()
    fdf['Risk Score'] = (fdf['_score']*100).round(1).astype(str)+'%'
    fdf['Status'] = '🚨 FRAUD'
    dcols = []
    if 'Time'   in fdf.columns: dcols.append('Time')
    if amt_col  in fdf.columns: dcols.append(amt_col)
    dcols += ['Risk Score','Status']
    st.dataframe(fdf[dcols].reset_index(drop=True), use_container_width=True,
                 height=min(320, 55+38*n_fraud))

# ALL TRANSACTIONS
st.markdown('<div class="section-hdr">All Transactions</div>', unsafe_allow_html=True)
ldf = df_res.copy()
ldf['Risk Score'] = (ldf['_score']*100).round(1).astype(str)+'%'
ldf['Status']     = ldf['_pred'].map({0:'✅ Safe', 1:'🚨 Fraud'})
scols = []
if 'Time'  in ldf.columns: scols.append('Time')
if amt_col: scols.append(amt_col)
scols += ['Risk Score','Status']
st.dataframe(ldf[scols].reset_index(drop=True), use_container_width=True, height=360)

# AMOUNT SCATTER 
if amt_col:
    st.markdown('<div class="section-hdr">Transaction Amounts vs Risk Score</div>', unsafe_allow_html=True)
    fig_sc = go.Figure()
    for lbl, mask, color, sym in [('Safe',fraud_pred==0,'#00cc66','circle'),('Fraud',fraud_pred==1,'#ff3c3c','star')]:
        if mask.any():
            fig_sc.add_trace(go.Scatter(
                x=df_res[mask].index, y=df_res[mask][amt_col], mode='markers', name=lbl,
                marker=dict(color=color, size=7 if lbl=='Safe' else 12, opacity=0.75, symbol=sym),
                text=(df_res[mask]['_score']*100).round(1).astype(str)+'%',
                hovertemplate='Amount: %{y}<br>Risk: %{text}<extra></extra>'
            ))
    fig_sc.update_layout(paper_bgcolor='#080e1e', plot_bgcolor='#080e1e',
        font_color='white', height=280, margin=dict(t=10,b=40,l=40,r=10),
        xaxis_title='Transaction #', yaxis_title='Amount (₹)',
        xaxis=dict(gridcolor='#1a2a4a'), yaxis=dict(gridcolor='#1a2a4a'),
        legend=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig_sc, use_container_width=True)

#  DOWNLOAD 
st.markdown('<div class="section-hdr">Download Report</div>', unsafe_allow_html=True)
st.download_button(
    label="⬇️ Download Fraud Detection Report (CSV)",
    data=ldf[scols].to_csv(index=False),
    file_name="creditguard_report.csv",
    mime="text/csv"
)

st.markdown("""<br>
<div style="text-align:center;color:#2a3a55;font-family:'JetBrains Mono',monospace;font-size:0.7em;letter-spacing:3px;padding-bottom:24px">
CREDITGUARD AI · FRAUD DETECTION ENGINE · ML LAB PROJECT
</div>""", unsafe_allow_html=True)
