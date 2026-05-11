import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    accuracy_score, f1_score, roc_auc_score
)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hotel Booking Cancellation Prediction",
    page_icon="🏨",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #0D1B3E; }
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }
[data-testid="metric-container"] {
    background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px;
}
.stButton > button {
    background: #1A56DB; color: white; border: none;
    border-radius: 8px; font-weight: 600; width: 100%;
}
.stButton > button:hover { background: #1347C8; }
.result-box {
    border-radius: 12px; padding: 24px; text-align: center;
    border: 1px solid #E2E8F0; background: white;
}
.big-prob { font-size: 52px; font-weight: 700; font-family: monospace; }
.badge-high   { background:#FEE2E2; color:#991B1B; padding:4px 14px; border-radius:20px; font-weight:600; font-size:13px; }
.badge-medium { background:#FEF3C7; color:#92400E; padding:4px 14px; border-radius:20px; font-weight:600; font-size:13px; }
.badge-low    { background:#D1FAE5; color:#065F46; padding:4px 14px; border-radius:20px; font-weight:600; font-size:13px; }
.info  { background:#EFF6FF; border-left:3px solid #1A56DB; padding:10px 14px; border-radius:0 8px 8px 0; font-size:13px; margin:6px 0; }
.warn  { background:#FFF7ED; border-left:3px solid #D97706; padding:10px 14px; border-radius:0 8px 8px 0; font-size:13px; margin:6px 0; }
.good  { background:#F0FDF4; border-left:3px solid #059669; padding:10px 14px; border-radius:0 8px 8px 0; font-size:13px; margin:6px 0; }
.alert { background:#FEF2F2; border-left:3px solid #DC2626; padding:10px 14px; border-radius:0 8px 8px 0; font-size:13px; margin:6px 0; }
</style>
""", unsafe_allow_html=True)

# ── Load data & models ─────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    data     = pd.read_csv("encode_hotel.csv")
    X_test   = joblib.load("X_test.pkl")
    y_test   = joblib.load("y_test.pkl")
    cb_model = joblib.load("catboost_model.pkl")

    scaler = joblib.load("scaler.pkl")
    X_test_scale = scaler.transform(X_test)
    return data, X_test_scale, y_test, cb_model, scaler

data, X_test_scale, y_test, cb_model, scaler = load_assets()

# ── Pre-compute test predictions ───────────────────────────────────────────
@st.cache_data
def get_test_results(_model, _X_test, _y_test):
    y_pred = _model.predict(_X_test)
    y_prob = _model.predict_proba(_X_test)[:, 1]
    return y_pred, y_prob

y_pred, y_prob = get_test_results(cb_model, X_test_scale, y_test)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Hotel Cancellation Predictor")
    st.markdown("### CatBoost Predictor")
    st.markdown("---")
    st.markdown("**Model files loaded:**")
    st.markdown("✅ `encode_hotel.csv`")
    st.markdown("✅ `catboost_model.pkl`")
    st.markdown("✅ `x_test.pkl`")
    st.markdown("✅ `y_test.pkl`")
    st.markdown("---")
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    f1  = f1_score(y_test, y_pred)
    st.markdown("**Model Performance**")
    st.markdown(f"- Accuracy : `{acc:.4f}`")
    st.markdown(f"- ROC-AUC  : `{roc:.4f}`")
    st.markdown(f"- F1 Score : `{f1:.4f}`")

# ── Title ──────────────────────────────────────────────────────────────────
st.title("Hotel Booking Cancellation Prediction")
st.markdown("Powered by **CatBoost** — fill in the booking details below or review model evaluation.")
st.markdown("---")

tab_predict, tab_eval = st.tabs(["🔮 Predict", "📊 Model Evaluation"])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("### Enter Booking Details")

    # ── Input form ─────────────────────────────────────────────────────────
    with st.form("booking_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Guest**")
            hotel         = st.selectbox("Hotel",          ["City Hotel", "Resort Hotel"])
            country       = st.selectbox("Country",        [
                "PRT","GBR","FRA","ESP","DEU","IRL","ITA","BEL","BRA","NLD",
                "USA","CHN","RUS","POL","ROU","AUT","CHE","SWE","NOR","DNK",
                "AUS","IND","JPN","KOR","TUR","MAR","AGO","ZAF","Other"
            ])
            is_repeat     = st.selectbox("Repeat guest?",  ["No", "Yes"])
            customer_type = st.selectbox("Customer type",
                ["Transient", "Transient-Party", "Contract", "Group"])
            adults        = st.number_input("Adults",   1, 10, 2)
            children      = st.number_input("Children", 0, 5,  0)
            babies        = st.number_input("Babies",   0, 3,  0)

        with col2:
            st.markdown("**Booking**")
            lead_time     = st.slider("Lead time (days)",     0, 350, 30)
            arrival_month = st.selectbox("Arrival month", [
                "January","February","March","April","May","June",
                "July","August","September","October","November","December"
            ])
            stays_nights  = st.number_input("Week nights",    1, 10,  2)
            meal          = st.selectbox("Meal plan",
                ["BB", "HB", "FB", "SC", "Undefined"])
            deposit_type  = st.selectbox("Deposit type",
                ["No Deposit", "Non Refund", "Refundable"])
            special_reqs  = st.slider("Special requests", 0, 5, 0)

        with col3:
            st.markdown("**Channel & Room**")
            market_seg    = st.selectbox("Market segment",
                ["Online TA","Offline TA/TO","Direct","Corporate",
                 "Groups","Complementary","Aviation"])
            dist_channel  = st.selectbox("Distribution channel",
                ["TA/TO","Direct","Corporate","GDS","Undefined"])
            reserved_room = st.selectbox("Reserved room type",
                ["A","B","C","D","E","F","G","H","L"])
            assigned_room = st.selectbox("Assigned room type",
                ["A","B","C","D","E","F","G","H","I","K","L"])
            adr           = st.number_input("Daily rate (€)", 0.0, 600.0, 100.0, step=5.0)
            booking_chg   = st.number_input("Booking changes",       0, 10, 0)
            prev_cancel   = st.number_input("Prev. cancellations",   0, 20, 0)

        predict_btn = st.form_submit_button("🔮 Predict Now", use_container_width=True)

    # ── On predict ─────────────────────────────────────────────────────────
    if predict_btn:

        # Derived features
        month_num   = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"
                       ].index(arrival_month) + 1
        has_sp      = 1 if special_reqs > 0 else 0
        match_rm    = 1 if reserved_room == assigned_room else 0
        hi_risk     = int(lead_time > 100 and deposit_type == "No Deposit" and has_sp == 0)
        total_g     = adults + children + babies
        lt          = lead_time
        lt_enc      = 0 if lt<=7 else 1 if lt<=30 else 2 if lt<=90 else 3 if lt<=180 else 4
        lt_bucket   = ("Last_minute" if lt<=7 else "Short" if lt<=30 else
                       "Medium"      if lt<=90 else "Long"  if lt<=180 else "Very_long")
        arr_q       = (month_num - 1) // 3 + 1

        # ── 245-column feature vector (same order as training)
        ALL_COLS = [
            'hotel','lead_time','stays_in_week_nights','adults','children','babies',
            'is_repeated_guest','previous_cancellations','previous_bookings_not_canceled',
            'booking_changes','days_in_waiting_list','adr','required_car_parking_spaces',
            'total_of_special_requests','arrival_year','arrival_month','arrival_quarter',
            'arrival_day_of_week','total_nights','total_guests','match_room_type',
            'has_special_request','is_high_risk','lead_time_bucket_encoded',
            'meal_BB','meal_FB','meal_HB','meal_SC','meal_Undefined',
            'market_segment_Aviation','market_segment_Complementary','market_segment_Corporate',
            'market_segment_Direct','market_segment_Groups','market_segment_Offline TA/TO',
            'market_segment_Online TA',
            'distribution_channel_Corporate','distribution_channel_Direct',
            'distribution_channel_GDS','distribution_channel_TA/TO','distribution_channel_Undefined',
            'deposit_type_No Deposit','deposit_type_Non Refund','deposit_type_Refundable',
            'customer_type_Contract','customer_type_Group','customer_type_Transient',
            'customer_type_Transient-Party',
            'reserved_room_type_A','reserved_room_type_B','reserved_room_type_C',
            'reserved_room_type_D','reserved_room_type_E','reserved_room_type_F',
            'reserved_room_type_G','reserved_room_type_H','reserved_room_type_L',
            'assigned_room_type_A','assigned_room_type_B','assigned_room_type_C',
            'assigned_room_type_D','assigned_room_type_E','assigned_room_type_F',
            'assigned_room_type_G','assigned_room_type_H','assigned_room_type_I',
            'assigned_room_type_K','assigned_room_type_L',
        ]

        COUNTRY_COLS = [
            'country_ABW','country_AGO','country_AIA','country_ALB','country_AND',
            'country_ARE','country_ARG','country_ARM','country_ASM','country_ATA',
            'country_AUS','country_AUT','country_AZE','country_BDI','country_BEL',
            'country_BEN','country_BFA','country_BGD','country_BGR','country_BHR',
            'country_BHS','country_BIH','country_BLR','country_BOL','country_BRA',
            'country_BRB','country_BWA','country_CAF','country_CHE','country_CHL',
            'country_CHN','country_CIV','country_CMR','country_CN','country_COL',
            'country_COM','country_CPV','country_CRI','country_CUB','country_CYM',
            'country_CYP','country_CZE','country_DEU','country_DJI','country_DMA',
            'country_DNK','country_DOM','country_DZA','country_ECU','country_EGY',
            'country_ESP','country_EST','country_ETH','country_FIN','country_FJI',
            'country_FRA','country_FRO','country_GAB','country_GBR','country_GEO',
            'country_GGY','country_GHA','country_GIB','country_GLP','country_GNB',
            'country_GRC','country_GTM','country_GUY','country_HKG','country_HND',
            'country_HRV','country_HUN','country_IDN','country_IMN','country_IND',
            'country_IRL','country_IRN','country_IRQ','country_ISL','country_ISR',
            'country_ITA','country_JAM','country_JEY','country_JOR','country_JPN',
            'country_KAZ','country_KEN','country_KHM','country_KIR','country_KNA',
            'country_KOR','country_KWT','country_LAO','country_LBN','country_LBY',
            'country_LCA','country_LIE','country_LKA','country_LTU','country_LUX',
            'country_LVA','country_MAC','country_MAR','country_MCO','country_MDV',
            'country_MEX','country_MKD','country_MLI','country_MLT','country_MMR',
            'country_MNE','country_MOZ','country_MRT','country_MUS','country_MWI',
            'country_MYS','country_MYT','country_NAM','country_NCL','country_NGA',
            'country_NIC','country_NLD','country_NOR','country_NPL','country_NZL',
            'country_OMN','country_PAK','country_PAN','country_PER','country_PHL',
            'country_PLW','country_POL','country_PRI','country_PRT','country_PRY',
            'country_PYF','country_QAT','country_ROU','country_RUS','country_RWA',
            'country_SAU','country_SDN','country_SEN','country_SGP','country_SLE',
            'country_SLV','country_SMR','country_SRB','country_STP','country_SUR',
            'country_SVK','country_SVN','country_SWE','country_SYC','country_SYR',
            'country_TGO','country_THA','country_TJK','country_TMP','country_TUN',
            'country_TUR','country_TWN','country_TZA','country_UGA','country_UKR',
            'country_UMI','country_URY','country_USA','country_UZB','country_VEN',
            'country_VGB','country_VNM','country_ZAF','country_ZMB','country_ZWE',
        ]

        RES_COLS    = ['reservation_status_Canceled','reservation_status_Check-Out',
                       'reservation_status_No-Show']
        BUCKET_COLS = ['lead_time_bucket_Last_minute','lead_time_bucket_Long',
                       'lead_time_bucket_Medium','lead_time_bucket_Short',
                       'lead_time_bucket_Very_long']

        ALL_245 = ALL_COLS + COUNTRY_COLS + RES_COLS + BUCKET_COLS

        row = {col: 0 for col in ALL_245}

        def s(k, v):
            if k in row: row[k] = v

        s("hotel",                          2 if hotel == "City Hotel" else 1)
        s("lead_time",                      lead_time)
        s("stays_in_week_nights",           stays_nights)
        s("adults",                         adults)
        s("children",                       children)
        s("babies",                         babies)
        s("is_repeated_guest",              1 if is_repeat == "Yes" else 0)
        s("previous_cancellations",         prev_cancel)
        s("previous_bookings_not_canceled", 0)
        s("booking_changes",                booking_chg)
        s("days_in_waiting_list",           0)
        s("adr",                            adr)
        s("required_car_parking_spaces",    0)
        s("total_of_special_requests",      special_reqs)
        s("arrival_year",                   2024)
        s("arrival_month",                  month_num)
        s("arrival_quarter",                arr_q)
        s("arrival_day_of_week",            0)
        s("total_nights",                   stays_nights)
        s("total_guests",                   total_g)
        s("match_room_type",                match_rm)
        s("has_special_request",            has_sp)
        s("is_high_risk",                   hi_risk)
        s("lead_time_bucket_encoded",       lt_enc)
        s(f"meal_{meal}",                   1)
        s(f"market_segment_{market_seg}",   1)
        s(f"distribution_channel_{dist_channel}", 1)
        s({"No Deposit":"deposit_type_No Deposit",
           "Non Refund":"deposit_type_Non Refund",
           "Refundable":"deposit_type_Refundable"}[deposit_type], 1)
        s(f"customer_type_{customer_type}", 1)
        s(f"reserved_room_type_{reserved_room}", 1)
        s(f"assigned_room_type_{assigned_room}", 1)
        if country != "Other":
            s(f"country_{country}", 1)
        s("reservation_status_Check-Out",   1)
        s(f"lead_time_bucket_{lt_bucket}",  1)

        input_arr        = np.array([[row[c] for c in ALL_245]], dtype=np.float64)
        input_arr_scaled = scaler.transform(input_arr)
        prob             = float(cb_model.predict_proba(input_arr_scaled)[0, 1])

        # Risk band
        if prob >= 0.65:
            risk_lbl, risk_cls, risk_clr, risk_ico = "HIGH RISK",   "badge-high",   "#DC2626", "🔴"
        elif prob >= 0.35:
            risk_lbl, risk_cls, risk_clr, risk_ico = "MEDIUM RISK", "badge-medium", "#D97706", "🟡"
        else:
            risk_lbl, risk_cls, risk_clr, risk_ico = "LOW RISK",    "badge-low",    "#059669", "🟢"

        st.markdown("---")
        st.markdown("### 📋 Prediction Result")

        res1, res2, res3 = st.columns([1, 1, 2])

        # ── Probability card
        with res1:
            st.markdown(f"""
            <div class="result-box">
                <div style="font-size:13px;color:#94A3B8;margin-bottom:6px">
                    Cancellation Probability
                </div>
                <div class="big-prob" style="color:{risk_clr}">{prob*100:.1f}%</div>
                <br>
                <span class="{risk_cls}">{risk_ico} {risk_lbl}</span>
                <div style="font-size:12px;color:#94A3B8;margin-top:10px">
                    {"Will likely cancel" if prob >= 0.5 else "Will likely stay"}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Gauge
        with res2:
            fig_g, ax_g = plt.subplots(figsize=(3.5, 3.5), facecolor="white")
            theta = np.linspace(np.pi, 0, 300)
            ax_g.plot(np.cos(theta), np.sin(theta), lw=18, color="#F1F5F9", solid_capstyle="butt")
            filled = np.linspace(np.pi, np.pi - prob * np.pi, 300)
            ax_g.plot(np.cos(filled), np.sin(filled), lw=18, color=risk_clr, solid_capstyle="butt")
            for pct, lbl in [(0,"0%"),(0.5,"50%"),(1,"100%")]:
                ang = np.pi - pct * np.pi
                ax_g.text(1.25*np.cos(ang), 1.25*np.sin(ang), lbl,
                          ha="center", va="center", fontsize=9, color="#94A3B8")
            ax_g.text(0, -0.05, f"{prob*100:.1f}%",
                      ha="center", va="center", fontsize=26, fontweight="bold",
                      color=risk_clr, fontfamily="monospace")
            ax_g.set_xlim(-1.4,1.4); ax_g.set_ylim(-0.3,1.3)
            ax_g.set_aspect("equal"); ax_g.axis("off")
            st.pyplot(fig_g, use_container_width=False); plt.close()

        # ── Risk signals + action
        with res3:
            st.markdown("**Risk signals detected:**")

            signals = []
            if lead_time > 100:
                signals.append(("alert", f"🔴 Lead time {lead_time} days — long-advance bookings cancel 2× more"))
            if deposit_type == "Non Refund":
                signals.append(("alert", "🔴 Non-refundable deposit — 94.6% cancel rate in dataset"))
            if deposit_type == "No Deposit":
                signals.append(("warn",  "🟡 No deposit — low financial commitment"))
            if prev_cancel > 0:
                signals.append(("alert", f"🔴 {prev_cancel} prior cancellation(s) on record"))
            if market_seg in ("Online TA","Complementary"):
                signals.append(("warn",  f"🟡 {market_seg} — above-average cancel rate channel"))
            if special_reqs == 0:
                signals.append(("warn",  "🟡 No special requests — lower intent signal"))
            if match_rm == 0:
                signals.append(("warn",  "🟡 Room type mismatch — guest may have preference issues"))
            if booking_chg > 0:
                signals.append(("good",  f"🟢 {booking_chg} booking change(s) — shows active engagement"))
            if special_reqs >= 2:
                signals.append(("good",  f"🟢 {special_reqs} special requests — strong intent to stay"))
            if is_repeat == "Yes":
                signals.append(("good",  "🟢 Repeat guest — cancels 3× less than new guests"))
            if lead_time <= 7:
                signals.append(("good",  "🟢 Last-minute booking — very low cancel rate (9%)"))

            for css_cls, msg in signals:
                st.markdown(f'<div class="{css_cls}">{msg}</div>', unsafe_allow_html=True)

            st.markdown("**Recommended action:**")
            if prob >= 0.65:
                st.markdown('<div class="alert">📧 Send re-confirmation email. Consider requesting deposit or offering loyalty incentive.</div>',
                            unsafe_allow_html=True)
            elif prob >= 0.35:
                st.markdown('<div class="warn">👀 Monitor this booking. Follow up 30 days before arrival.</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="good">✅ Low risk — no intervention needed.</div>',
                            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("### Model Evaluation on Test Set")

    # ── Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",    f"{accuracy_score(y_test, y_pred):.4f}")
    m2.metric("ROC-AUC",     f"{roc_auc_score(y_test, y_prob):.4f}")
    m3.metric("F1 Score",    f"{f1_score(y_test, y_pred):.4f}")
    m4.metric("Test Samples",f"{len(y_test):,}")

    st.markdown("---")
    c1, c2 = st.columns(2)

    # ── ROC Curve
    with c1:
        st.markdown("#### ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc_val = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor="white")
        ax.plot(fpr, tpr, color="#1A56DB", lw=2.5,
                label=f"CatBoost (AUC = {roc_auc_val:.4f})")
        ax.fill_between(fpr, tpr, alpha=0.08, color="#1A56DB")
        ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.4, label="Random baseline")
        ax.set_xlabel("False Positive Rate", fontsize=10, color="#64748B")
        ax.set_ylabel("True Positive Rate",  fontsize=10, color="#64748B")
        ax.set_title("ROC Curve",            fontsize=12, color="#1E293B")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.set_facecolor("white")
        st.pyplot(fig); plt.close()

    # ── Confusion Matrix
    with c2:
        st.markdown("#### Confusion Matrix")
        cm     = confusion_matrix(y_test, y_pred)
        cm_pct = cm / cm.sum(axis=1, keepdims=True)

        fig2, ax2 = plt.subplots(figsize=(5, 4.2), facecolor="white")
        sns.heatmap(cm, annot=False, cmap="Blues", ax=ax2,
                    xticklabels=["Stayed","Cancelled"],
                    yticklabels=["Stayed","Cancelled"],
                    linewidths=0.5, cbar=False)
        for i in range(2):
            for j in range(2):
                dark = cm_pct[i,j] > 0.5
                ax2.text(j+0.5, i+0.38, f"{cm[i,j]:,}",
                         ha="center", va="center", fontsize=16, fontweight="bold",
                         color="white" if dark else "#1E293B")
                ax2.text(j+0.5, i+0.68, f"({cm_pct[i,j]:.1%})",
                         ha="center", va="center", fontsize=10,
                         color="white" if dark else "#64748B")
        ax2.set_xlabel("Predicted", color="#64748B", fontsize=10)
        ax2.set_ylabel("Actual",    color="#64748B", fontsize=10)
        ax2.set_title("Confusion Matrix", color="#1E293B", fontsize=12)
        ax2.set_facecolor("white")
        st.pyplot(fig2); plt.close()

    st.markdown("---")
    c3, c4 = st.columns(2)

    # ── Score distribution
    with c3:
        st.markdown("#### Predicted Probability Distribution")
        fig3, ax3 = plt.subplots(figsize=(5.5, 4), facecolor="white")
        ax3.hist(y_prob[y_test==0], bins=50, alpha=0.65, color="#0891B2",
                 label="Stayed",    density=True)
        ax3.hist(y_prob[y_test==1], bins=50, alpha=0.65, color="#DC2626",
                 label="Cancelled", density=True)
        ax3.axvline(0.5, color="black", lw=1.5, ls="--", label="Threshold 0.5")
        ax3.set_xlabel("Predicted Probability", color="#64748B", fontsize=10)
        ax3.set_ylabel("Density",               color="#64748B", fontsize=10)
        ax3.set_title("Score Distribution by Class", color="#1E293B", fontsize=12)
        ax3.legend(fontsize=9)
        ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
        ax3.set_facecolor("white")
        st.pyplot(fig3); plt.close()

    # ── Classification report
    with c4:
        st.markdown("#### Classification Report")
        report = classification_report(
            y_test, y_pred,
            target_names=["Stayed", "Cancelled"],
            output_dict=True
        )
        rep_df = pd.DataFrame({
            k: v for k, v in report.items()
            if isinstance(v, dict) and k in
               ("Stayed", "Cancelled", "macro avg", "weighted avg")
        }).T.round(4)
        st.dataframe(rep_df, use_container_width=True)
