import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HotelOps AI · Booking Analytics",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #f8f9fa; }
.metric-card {
    background: #ffffff;
    border: 1px solid #e9ecef;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
}
.risk-high   { color: #dc3545; font-weight: 600; }
.risk-medium { color: #fd7e14; font-weight: 600; }
.risk-low    { color: #198754; font-weight: 600; }
.rec-urgent  { background:#fff5f5; border-left:4px solid #dc3545; padding:0.75rem 1rem; border-radius:0 8px 8px 0; margin-bottom:0.5rem; }
.rec-medium  { background:#fff8f0; border-left:4px solid #fd7e14; padding:0.75rem 1rem; border-radius:0 8px 8px 0; margin-bottom:0.5rem; }
.rec-low     { background:#f0fff4; border-left:4px solid #198754; padding:0.75rem 1rem; border-radius:0 8px 8px 0; margin-bottom:0.5rem; }
h1 { font-size: 1.6rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'hotel','lead_time','stays_in_week_nights','adults','children','babies',
    'meal','market_segment','distribution_channel','is_repeated_guest',
    'previous_cancellations','previous_bookings_not_canceled',
    'reserved_room_type','assigned_room_type','booking_changes',
    'deposit_type','days_in_waiting_list','customer_type','adr',
    'required_car_parking_spaces','total_of_special_requests',
    'total_stay','total_guests','is_modified_booking','high_lead_time','month'
]
CAT_COLS = ['hotel','meal','market_segment','distribution_channel',
            'reserved_room_type','assigned_room_type','deposit_type','customer_type','month']

HOTEL_OPTIONS        = ['City Hotel', 'Resort Hotel']
MEAL_OPTIONS         = ['BB', 'FB', 'HB', 'SC', 'Undefined']
MARKET_OPTIONS       = ['Aviation','Complementary','Corporate','Direct','Groups',
                        'Offline TA/TO','Online TA','Undefined']
DISTRIBUTION_OPTIONS = ['Corporate','Direct','GDS','TA/TO','Undefined']
ROOM_OPTIONS         = ['A','B','C','D','E','F','G','H','L','P']
ASSIGNED_OPTIONS     = ['A','B','C','D','E','F','G','H','I','K','L','P']
DEPOSIT_OPTIONS      = ['No Deposit','Non Refund','Refundable']
CUSTOMER_OPTIONS     = ['Contract','Group','Transient','Transient-Party']
MONTH_OPTIONS        = ['January','February','March','April','May','June',
                        'July','August','September','October','November','December']

MONTH_ORDER = MONTH_OPTIONS

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("hotel_model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("combined_hotel_dataset.csv")
    df['children'] = df['children'].fillna(0)
    df['country']  = df['country'].fillna('Unknown')
    df['agent']    = df['agent'].fillna(0)
    df['total_stay']          = df['stays_in_week_nights']
    df['total_guests']        = df['adults'] + df['children'] + df['babies']
    df['is_modified_booking'] = (df['booking_changes'] > 0).astype(int)
    df['high_lead_time']      = (df['lead_time'] > 90).astype(int)
    df['month']               = pd.to_datetime(df['arrival_date']).dt.strftime('%B')
    df['arrival_dt']          = pd.to_datetime(df['arrival_date'])
    df['year']                = df['arrival_dt'].dt.year
    for c in CAT_COLS:
        df[c] = df[c].fillna('Unknown').astype(str)
    return df

def predict_cancel_prob(model, row_dict):
    """Given a dict of raw booking fields, return cancel probability."""
    df_input = pd.DataFrame([row_dict])
    df_input['total_stay']          = df_input['stays_in_week_nights']
    df_input['total_guests']        = df_input['adults'] + df_input['children'] + df_input['babies']
    df_input['is_modified_booking'] = (df_input['booking_changes'] > 0).astype(int)
    df_input['high_lead_time']      = (df_input['lead_time'] > 90).astype(int)
    for c in CAT_COLS:
        df_input[c] = df_input[c].fillna('Unknown').astype(str)
    X = df_input[FEATURE_COLS]
    return model.predict_proba(X)[0][1]

def risk_label(prob):
    if prob >= 0.65: return "🔴 High Risk", "risk-high"
    if prob >= 0.40: return "🟠 Medium Risk", "risk-medium"
    return "🟢 Low Risk", "risk-low"

# ── Sidebar nav ───────────────────────────────────────────────────────────────
st.sidebar.title("HotelOps AI")
st.sidebar.caption("Powered by Logistic Regression · 80% accuracy")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Overview Dashboard",
     "🔍 Booking Risk Scorer",
     "👥 Guest Segment Analysis",
     "📅 Demand & Seasonality",
     "🍽️ Meal & Operations",
     "💡 What-If Simulator"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.caption("Dataset: 119,390 bookings\nCity Hotel + Resort Hotel")

# ── Load ──────────────────────────────────────────────────────────────────────
model = load_model()
df    = load_data()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 · OVERVIEW DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview Dashboard":
    st.title("📊 Operational Overview")
    st.caption("Historical booking data — City Hotel & Resort Hotel combined")

    # ── KPI row ──
    total       = len(df)
    cancelled   = df['is_canceled'].sum()
    cancel_rate = cancelled / total
    avg_adr     = df['adr'].clip(0).mean()
    avg_lead    = df['lead_time'].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Bookings",      f"{total:,}")
    c2.metric("Cancellation Rate",   f"{cancel_rate:.1%}",   delta="-63% completed", delta_color="inverse")
    c3.metric("Avg Daily Rate (ADR)", f"RM {avg_adr:.0f}")
    c4.metric("Avg Lead Time",        f"{avg_lead:.0f} days")

    st.markdown("---")

    col_l, col_r = st.columns(2)

    # ── Cancellation by hotel type ──
    with col_l:
        st.subheader("Cancellation Rate by Hotel Type")
        hotel_stats = df.groupby('hotel')['is_canceled'].agg(['sum','count']).reset_index()
        hotel_stats['rate'] = hotel_stats['sum'] / hotel_stats['count']
        fig = px.bar(hotel_stats, x='hotel', y='rate', color='hotel',
                     color_discrete_map={'City Hotel':'#E24B4A','Resort Hotel':'#3B8BD4'},
                     labels={'rate':'Cancellation Rate','hotel':'Hotel Type'},
                     text=hotel_stats['rate'].apply(lambda x: f"{x:.1%}"))
        fig.update_traces(textposition='outside')
        fig.update_layout(yaxis_tickformat='.0%', showlegend=False, height=320,
                          margin=dict(t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # ── Lead time distribution ──
    with col_r:
        st.subheader("Lead Time: Cancelled vs Completed")
        lead_cancel  = df[df['is_canceled']==1]['lead_time'].clip(0,500)
        lead_complete = df[df['is_canceled']==0]['lead_time'].clip(0,500)
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=lead_cancel,  name='Cancelled',  opacity=0.7,
                                    marker_color='#E24B4A', nbinsx=50))
        fig2.add_trace(go.Histogram(x=lead_complete, name='Completed',  opacity=0.7,
                                    marker_color='#3B8BD4', nbinsx=50))
        fig2.update_layout(barmode='overlay', height=320,
                           xaxis_title='Lead Time (days)', yaxis_title='Count',
                           margin=dict(t=20,b=0), legend=dict(x=0.7,y=0.9))
        st.plotly_chart(fig2, use_container_width=True)

    col_l2, col_r2 = st.columns(2)

    # ── Monthly bookings ──
    with col_l2:
        st.subheader("Monthly Booking Volume & ADR")
        monthly = df.groupby('month').agg(
            bookings=('is_canceled','count'),
            avg_adr=('adr','mean')
        ).reindex(MONTH_ORDER).reset_index()
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Bar(x=monthly['month'], y=monthly['bookings'],
                              name='Bookings', marker_color='#3B8BD4', opacity=0.8),
                       secondary_y=False)
        fig3.add_trace(go.Scatter(x=monthly['month'], y=monthly['avg_adr'],
                                  name='Avg ADR (RM)', mode='lines+markers',
                                  line=dict(color='#EF9F27', width=2)),
                       secondary_y=True)
        fig3.update_layout(height=320, margin=dict(t=20,b=0),
                           legend=dict(x=0.01,y=0.99))
        fig3.update_xaxes(tickangle=45)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Segment cancel rate ──
    with col_r2:
        st.subheader("Cancellation Rate by Customer Segment")
        seg = df.groupby('customer_type')['is_canceled'].mean().sort_values(ascending=True).reset_index()
        seg.columns = ['Segment','Cancel Rate']
        fig4 = px.bar(seg, x='Cancel Rate', y='Segment', orientation='h',
                      color='Cancel Rate',
                      color_continuous_scale=['#198754','#fd7e14','#dc3545'],
                      text=seg['Cancel Rate'].apply(lambda x: f"{x:.1%}"))
        fig4.update_traces(textposition='outside')
        fig4.update_layout(height=320, coloraxis_showscale=False,
                           margin=dict(t=20,b=0),
                           xaxis_tickformat='.0%')
        st.plotly_chart(fig4, use_container_width=True)

    # ── Insights box ──
    st.markdown("---")
    st.subheader("📌 Key Operational Insights")
    city_rate   = df[df['hotel']=='City Hotel']['is_canceled'].mean()
    resort_rate = df[df['hotel']=='Resort Hotel']['is_canceled'].mean()
    avg_lead_cancel   = df[df['is_canceled']==1]['lead_time'].mean()
    avg_lead_complete = df[df['is_canceled']==0]['lead_time'].mean()
    repeat_cancel = df[df['is_repeated_guest']==1]['is_canceled'].mean()
    new_cancel    = df[df['is_repeated_guest']==0]['is_canceled'].mean()

    i1, i2, i3 = st.columns(3)
    i1.info(f"**City Hotel** cancels at **{city_rate:.1%}** vs Resort at **{resort_rate:.1%}** — {(city_rate/resort_rate - 1):.0%} higher operational risk")
    i2.warning(f"Cancelled bookings were made **{avg_lead_cancel:.0f} days** in advance vs {avg_lead_complete:.0f} days for completed stays")
    i3.success(f"Repeat guests cancel at only **{repeat_cancel:.1%}** vs **{new_cancel:.1%}** for new guests — loyalty pays off")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 · BOOKING RISK SCORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Booking Risk Scorer":
    st.title("🔍 Booking Cancellation Risk Scorer")
    st.caption("Enter booking details below to get an AI-powered cancellation risk score")

    with st.form("booking_form"):
        st.subheader("Booking Details")
        r1c1, r1c2, r1c3 = st.columns(3)
        hotel       = r1c1.selectbox("Hotel Type",         HOTEL_OPTIONS)
        customer    = r1c2.selectbox("Customer Type",      CUSTOMER_OPTIONS)
        market      = r1c3.selectbox("Market Segment",     MARKET_OPTIONS)

        r2c1, r2c2, r2c3 = st.columns(3)
        distribution = r2c1.selectbox("Distribution Channel", DISTRIBUTION_OPTIONS)
        deposit      = r2c2.selectbox("Deposit Type",         DEPOSIT_OPTIONS)
        meal         = r2c3.selectbox("Meal Package",         MEAL_OPTIONS)

        r3c1, r3c2, r3c3, r3c4 = st.columns(4)
        lead_time   = r3c1.number_input("Lead Time (days)", 0, 737, 60)
        week_nights = r3c2.number_input("Week Nights",       0, 20,  2)
        adults      = r3c3.number_input("Adults",            1, 10,  2)
        children    = r3c4.number_input("Children",          0, 10,  0)

        r4c1, r4c2, r4c3, r4c4 = st.columns(4)
        babies          = r4c1.number_input("Babies",           0, 5,  0)
        adr             = r4c2.number_input("ADR (RM)",         0, 5400, 120)
        special_req     = r4c3.number_input("Special Requests", 0, 10, 0)
        booking_changes = r4c4.number_input("Booking Changes",  0, 20, 0)

        r5c1, r5c2, r5c3, r5c4 = st.columns(4)
        prev_cancel  = r5c1.number_input("Previous Cancellations",     0, 26, 0)
        prev_ok      = r5c2.number_input("Previous Completed Stays",   0, 72, 0)
        parking      = r5c3.number_input("Parking Spaces Required",    0, 8,  0)
        waiting_days = r5c4.number_input("Days on Waiting List",       0, 391,0)

        r6c1, r6c2, r6c3, r6c4 = st.columns(4)
        reserved_room = r6c1.selectbox("Reserved Room",  ROOM_OPTIONS)
        assigned_room = r6c2.selectbox("Assigned Room",  ASSIGNED_OPTIONS)
        is_repeated   = r6c3.selectbox("Repeat Guest",   [0, 1], format_func=lambda x: "Yes" if x else "No")
        month         = r6c4.selectbox("Arrival Month",  MONTH_OPTIONS, index=6)

        submitted = st.form_submit_button("🔮 Predict Cancellation Risk", use_container_width=True)

    if submitted:
        booking = dict(
            hotel=hotel, lead_time=lead_time, stays_in_week_nights=week_nights,
            adults=adults, children=float(children), babies=babies, meal=meal,
            market_segment=market, distribution_channel=distribution,
            is_repeated_guest=is_repeated, previous_cancellations=prev_cancel,
            previous_bookings_not_canceled=prev_ok, reserved_room_type=reserved_room,
            assigned_room_type=assigned_room, booking_changes=booking_changes,
            deposit_type=deposit, days_in_waiting_list=waiting_days,
            customer_type=customer, adr=float(adr),
            required_car_parking_spaces=parking,
            total_of_special_requests=special_req, month=month,
        )
        prob = predict_cancel_prob(model, booking)
        label, css = risk_label(prob)

        st.markdown("---")
        rc1, rc2, rc3 = st.columns([1,2,1])
        with rc2:
            st.markdown(f"### Cancellation Probability")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                number={'suffix': '%'},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar':  {'color': "#E24B4A" if prob >= 0.65 else "#fd7e14" if prob >= 0.4 else "#198754"},
                    'steps': [
                        {'range': [0,  40], 'color': '#d4edda'},
                        {'range': [40, 65], 'color': '#fff3cd'},
                        {'range': [65,100], 'color': '#f8d7da'},
                    ],
                    'threshold': {'line': {'color': 'black', 'width': 2}, 'value': prob*100}
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=20,b=0,l=20,r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown(f"<div style='text-align:center;font-size:1.3rem' class='{css}'>{label}</div>",
                        unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("💡 Recommended Actions")

        recs = []
        if prob >= 0.65:
            recs.append(("urgent", "📧 Send reconfirmation email immediately",
                          "This booking has a high cancellation risk. A personalised reconfirmation can reduce cancel probability by up to 20%."))
            if deposit == "No Deposit":
                recs.append(("urgent", "💳 Consider requesting a deposit",
                              "Non-refundable or partial deposits significantly reduce cancellation likelihood for high-risk bookings."))
        if lead_time > 120:
            recs.append(("medium", "📅 Schedule a 60-day pre-arrival check-in",
                          f"With {lead_time} days lead time, customer plans may change. A mid-way reconfirmation reduces uncertainty."))
        if prev_cancel > 0:
            recs.append(("urgent", "⚠️ Guest has previous cancellation history",
                          f"{prev_cancel} prior cancellation(s) detected. Monitor closely and consider overbooking buffer."))
        if special_req == 0 and booking_changes == 0:
            recs.append(("medium", "🤝 Engage guest to boost commitment",
                          "Guests with zero special requests and no modifications cancel more often. Reach out to personalise their experience."))
        if parking == 0 and special_req == 0:
            recs.append(("low", "🎁 Offer a loyalty incentive",
                          "Low engagement signals. A complimentary upgrade or welcome gift can strengthen commitment."))
        if not recs:
            recs.append(("low", "✅ Booking looks stable",
                          "No immediate action needed. Maintain normal operations for this booking."))

        for urgency, title, detail in recs:
            st.markdown(f"""<div class='rec-{urgency}'>
                <strong>{title}</strong><br><small>{detail}</small>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 · GUEST SEGMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Guest Segment Analysis":
    st.title("👥 Guest Segment Analysis")
    st.caption("Understand which guest types drive cancellations and how to manage them")

    tab1, tab2, tab3 = st.tabs(["Customer Type", "Market Segment", "Deposit & Loyalty"])

    with tab1:
        ctype = df.groupby('customer_type').agg(
            bookings=('is_canceled','count'),
            cancellations=('is_canceled','sum'),
            avg_lead=('lead_time','mean'),
            avg_adr=('adr','mean'),
        ).reset_index()
        ctype['cancel_rate'] = ctype['cancellations'] / ctype['bookings']

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Cancellation Rate")
            fig = px.bar(ctype.sort_values('cancel_rate'), x='cancel_rate', y='customer_type',
                         orientation='h', color='cancel_rate',
                         color_continuous_scale=['#198754','#fd7e14','#dc3545'],
                         text=ctype.sort_values('cancel_rate')['cancel_rate'].apply(lambda x: f"{x:.1%}"))
            fig.update_traces(textposition='outside')
            fig.update_layout(height=300, coloraxis_showscale=False,
                              xaxis_tickformat='.0%', margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Booking Volume")
            fig2 = px.pie(ctype, names='customer_type', values='bookings',
                          color_discrete_sequence=px.colors.qualitative.Set2)
            fig2.update_layout(height=300, margin=dict(t=10))
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(ctype.rename(columns={
            'customer_type':'Segment','bookings':'Total Bookings',
            'cancellations':'Cancelled','cancel_rate':'Cancel Rate',
            'avg_lead':'Avg Lead Days','avg_adr':'Avg ADR (RM)'
        }).assign(**{
            'Cancel Rate': lambda d: d['Cancel Rate'].map('{:.1%}'.format),
            'Avg Lead Days': lambda d: d['Avg Lead Days'].map('{:.0f}'.format),
            'Avg ADR (RM)': lambda d: d['Avg ADR (RM)'].map('{:.0f}'.format),
        }), use_container_width=True, hide_index=True)

    with tab2:
        mseg = df.groupby('market_segment').agg(
            bookings=('is_canceled','count'),
            cancellations=('is_canceled','sum'),
            avg_lead=('lead_time','mean'),
        ).reset_index()
        mseg['cancel_rate'] = mseg['cancellations'] / mseg['bookings']
        mseg = mseg.sort_values('cancel_rate', ascending=False)

        fig3 = px.scatter(mseg, x='avg_lead', y='cancel_rate', size='bookings',
                          color='cancel_rate', text='market_segment',
                          color_continuous_scale=['#198754','#fd7e14','#dc3545'],
                          labels={'avg_lead':'Avg Lead Time (days)',
                                  'cancel_rate':'Cancellation Rate',
                                  'bookings':'Volume'})
        fig3.update_traces(textposition='top center')
        fig3.update_layout(yaxis_tickformat='.0%', height=420,
                           coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Bubble size = booking volume. Top-right = high risk, high lead time.")

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            dep = df.groupby('deposit_type')['is_canceled'].mean().reset_index()
            dep.columns = ['Deposit Type','Cancel Rate']
            fig4 = px.bar(dep, x='Deposit Type', y='Cancel Rate',
                          color='Cancel Rate',
                          color_continuous_scale=['#198754','#fd7e14','#dc3545'],
                          text=dep['Cancel Rate'].apply(lambda x: f"{x:.1%}"))
            fig4.update_traces(textposition='outside')
            fig4.update_layout(yaxis_tickformat='.0%', coloraxis_showscale=False,
                                height=320, title='Cancellation Rate by Deposit Type',
                                margin=dict(t=40))
            st.plotly_chart(fig4, use_container_width=True)
        with c2:
            rep = df.groupby('is_repeated_guest')['is_canceled'].mean().reset_index()
            rep['Guest Type'] = rep['is_repeated_guest'].map({0:'New Guest',1:'Repeat Guest'})
            fig5 = px.bar(rep, x='Guest Type', y='is_canceled',
                          color='is_canceled',
                          color_continuous_scale=['#198754','#dc3545'],
                          text=rep['is_canceled'].apply(lambda x: f"{x:.1%}"),
                          labels={'is_canceled':'Cancel Rate'})
            fig5.update_traces(textposition='outside')
            fig5.update_layout(yaxis_tickformat='.0%', coloraxis_showscale=False,
                                height=320, title='Repeat vs New Guest Cancel Rate',
                                margin=dict(t=40))
            st.plotly_chart(fig5, use_container_width=True)

        st.info("💡 **Operational takeaway:** Non-refundable deposits reduce cancellations by up to 95%. Loyalty programmes that convert new guests into repeat guests significantly stabilise operations.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 · DEMAND & SEASONALITY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📅 Demand & Seasonality":
    st.title("📅 Demand Patterns & Seasonality")

    hotel_filter = st.radio("Hotel Type", ["Both", "City Hotel", "Resort Hotel"], horizontal=True)
    dff = df if hotel_filter == "Both" else df[df['hotel'] == hotel_filter]

    # Monthly demand
    monthly = dff.groupby('month').agg(
        bookings=('is_canceled','count'),
        cancelled=('is_canceled','sum'),
        avg_adr=('adr','mean'),
        avg_lead=('lead_time','mean'),
    ).reindex(MONTH_ORDER).reset_index()
    monthly['completion_rate'] = (monthly['bookings'] - monthly['cancelled']) / monthly['bookings']

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=('Booking Volume & Cancellations', 'Average Daily Rate (RM)'),
                        vertical_spacing=0.12)
    fig.add_trace(go.Bar(x=monthly['month'], y=monthly['bookings'],
                         name='Total Bookings', marker_color='#3B8BD4', opacity=0.7), row=1, col=1)
    fig.add_trace(go.Bar(x=monthly['month'], y=monthly['cancelled'],
                         name='Cancelled', marker_color='#E24B4A', opacity=0.8), row=1, col=1)
    fig.add_trace(go.Scatter(x=monthly['month'], y=monthly['avg_adr'],
                             mode='lines+markers', name='Avg ADR',
                             line=dict(color='#EF9F27', width=2.5)), row=2, col=1)
    fig.update_layout(height=500, barmode='overlay', margin=dict(t=50,b=0))
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    # Staffing recommendation
    st.subheader("👷 Staffing & Capacity Planner")
    st.caption("Adjust assumptions below to get operational recommendations per month")

    col_s1, col_s2 = st.columns(2)
    staff_per_room = col_s1.slider("Staff needed per occupied room", 0.1, 1.0, 0.3, 0.05)
    buffer_pct     = col_s2.slider("Overbooking buffer (%)", 0, 30, 10)

    monthly['expected_arrivals'] = (monthly['bookings'] - monthly['cancelled']) * (1 + buffer_pct/100)
    monthly['recommended_staff'] = (monthly['expected_arrivals'] * staff_per_room).round(0).astype(int)

    fig2 = px.bar(monthly, x='month', y='recommended_staff',
                  color='recommended_staff',
                  color_continuous_scale=['#3B8BD4','#EF9F27','#E24B4A'],
                  text='recommended_staff',
                  labels={'recommended_staff':'Staff Needed','month':'Month'})
    fig2.update_traces(textposition='outside')
    fig2.update_layout(height=350, coloraxis_showscale=False, margin=dict(t=20))
    fig2.update_xaxes(tickangle=45)
    st.plotly_chart(fig2, use_container_width=True)

    peak_month = monthly.loc[monthly['recommended_staff'].idxmax(), 'month']
    low_month  = monthly.loc[monthly['recommended_staff'].idxmin(), 'month']
    st.success(f"📈 **Peak month:** {peak_month} — maximise staffing and inventory")
    st.info(f"📉 **Lowest demand:** {low_month} — opportunity to reduce costs and schedule maintenance")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 · MEAL & OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🍽️ Meal & Operations":
    st.title("🍽️ Meal Preferences & Operational Waste Reduction")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Meal Package Distribution")
        meal_map = {'BB':'Bed & Breakfast','FB':'Full Board','HB':'Half Board',
                    'SC':'Self Catering','Undefined':'Not Specified'}
        meal_counts = df['meal'].map(meal_map).value_counts().reset_index()
        meal_counts.columns = ['Meal','Count']
        fig = px.pie(meal_counts, names='Meal', values='Count',
                     color_discrete_sequence=['#3B8BD4','#EF9F27','#E24B4A','#198754','#6c757d'])
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Cancellation Rate by Meal Type")
        meal_cancel = df.groupby('meal')['is_canceled'].mean().reset_index()
        meal_cancel['meal_name'] = meal_cancel['meal'].map(meal_map)
        fig2 = px.bar(meal_cancel.sort_values('is_canceled'),
                      x='is_canceled', y='meal_name', orientation='h',
                      color='is_canceled',
                      color_continuous_scale=['#198754','#fd7e14','#dc3545'],
                      text=meal_cancel.sort_values('is_canceled')['is_canceled'].apply(lambda x: f"{x:.1%}"),
                      labels={'is_canceled':'Cancel Rate','meal_name':'Meal Type'})
        fig2.update_traces(textposition='outside')
        fig2.update_layout(height=350, coloraxis_showscale=False, xaxis_tickformat='.0%')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("🧮 Food Waste Estimator")
    st.caption("Estimate daily food waste cost based on meal demand and cancellations")

    co1, co2, co3 = st.columns(3)
    avg_daily_bookings = co1.number_input("Expected daily arrivals", 50, 500, 150)
    cost_per_meal      = co2.number_input("Avg cost per full-board meal (RM)", 10, 200, 45)
    overprep_pct       = co3.slider("Over-preparation buffer (%)", 0, 50, 20)

    bb_share  = (df['meal'] == 'BB').mean()
    fb_share  = (df['meal'] == 'FB').mean()
    hb_share  = (df['meal'] == 'HB').mean()
    cancel_rt = df['is_canceled'].mean()

    expected_lunch_dinner = avg_daily_bookings * (fb_share + hb_share * 0.5)
    wasted_meals = expected_lunch_dinner * cancel_rt * (1 + overprep_pct/100)
    waste_cost   = wasted_meals * cost_per_meal

    w1, w2, w3 = st.columns(3)
    w1.metric("Guests needing lunch/dinner", f"{expected_lunch_dinner:.0f}")
    w2.metric("Est. wasted meals/day",        f"{wasted_meals:.0f}")
    w3.metric("Est. daily waste cost",         f"RM {waste_cost:,.0f}")

    st.warning(f"💡 **Recommendation:** {bb_share:.1%} of guests choose BB only. Focus kitchen investment on breakfast. Reducing lunch/dinner prep by 30% could save approximately **RM {waste_cost*0.3*30:,.0f}/month**.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 · WHAT-IF SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💡 What-If Simulator":
    st.title("💡 What-If Revenue & Operations Simulator")
    st.caption("Simulate the financial impact of reducing cancellations through operational changes")

    st.subheader("Current Performance (Based on Dataset)")
    total_bookings = len(df)
    total_cancelled = df['is_canceled'].sum()
    avg_adr = df['adr'].clip(0).mean()
    avg_nights = df['stays_in_week_nights'].mean()
    base_revenue_lost = total_cancelled * avg_adr * avg_nights

    b1, b2, b3 = st.columns(3)
    b1.metric("Total Bookings",       f"{total_bookings:,}")
    b2.metric("Cancellations",        f"{total_cancelled:,} ({total_cancelled/total_bookings:.1%})")
    b3.metric("Est. Revenue Lost",    f"RM {base_revenue_lost:,.0f}")

    st.markdown("---")
    st.subheader("🔧 Adjust Operational Levers")

    lev1, lev2, lev3 = st.columns(3)
    reconfirm_rate  = lev1.slider("% of high-risk bookings that get reconfirmation emails", 0, 100, 50)
    reconfirm_eff   = lev1.slider("Effectiveness: cancel reduction per reconfirmation (%)", 0, 40, 15)
    loyalty_rate    = lev2.slider("% increase in repeat guest bookings", 0, 50, 10)
    deposit_rate    = lev3.slider("% of bookings moved to non-refundable deposit", 0, 50, 20)
    deposit_eff     = lev3.slider("Cancel reduction from deposits (%)", 0, 80, 60)

    # Compute
    high_risk_bookings = int(total_bookings * 0.35)
    saves_reconfirm = high_risk_bookings * (reconfirm_rate/100) * (reconfirm_eff/100)

    repeat_share_now  = df['is_repeated_guest'].mean()
    repeat_cancel_rt  = df[df['is_repeated_guest']==1]['is_canceled'].mean()
    new_cancel_rt     = df[df['is_repeated_guest']==0]['is_canceled'].mean()
    extra_repeat      = total_bookings * (loyalty_rate/100)
    saves_loyalty     = extra_repeat * (new_cancel_rt - repeat_cancel_rt)

    high_risk_no_dep  = int(total_bookings * 0.4)
    saves_deposit     = high_risk_no_dep * (deposit_rate/100) * (deposit_eff/100)

    total_saves = saves_reconfirm + saves_loyalty + saves_deposit
    new_cancel  = max(0, total_cancelled - total_saves)
    new_rate    = new_cancel / total_bookings
    revenue_saved = total_saves * avg_adr * avg_nights

    st.markdown("---")
    st.subheader("📈 Projected Outcome")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Cancellations prevented",    f"{total_saves:,.0f}")
    p2.metric("New cancellation rate",       f"{new_rate:.1%}", delta=f"{(new_rate - total_cancelled/total_bookings):.1%}")
    p3.metric("Revenue recovered",           f"RM {revenue_saved:,.0f}", delta="vs current")
    p4.metric("ROI on intervention costs",   f"{(revenue_saved/(total_bookings*2)):.1f}x")

    # Waterfall chart
    fig = go.Figure(go.Waterfall(
        name="Impact", orientation="v",
        measure=["absolute","relative","relative","relative","total"],
        x=["Current\nCancellations","Reconfirmation\nEmails",
           "Loyalty\nProgramme","Deposit\nPolicy","Projected\nCancellations"],
        y=[total_cancelled, -saves_reconfirm, -saves_loyalty, -saves_deposit, 0],
        connector={"line":{"color":"#adb5bd"}},
        decreasing={"marker":{"color":"#198754"}},
        increasing={"marker":{"color":"#E24B4A"}},
        totals={"marker":{"color":"#3B8BD4"}},
        text=[f"{total_cancelled:,.0f}",
              f"-{saves_reconfirm:,.0f}",
              f"-{saves_loyalty:,.0f}",
              f"-{saves_deposit:,.0f}",
              f"{new_cancel:,.0f}"],
        textposition="outside"
    ))
    fig.update_layout(height=420, title="Cancellation Reduction Waterfall",
                      margin=dict(t=50))
    st.plotly_chart(fig, use_container_width=True)

    st.success(f"""
    🎯 **Summary:** By implementing reconfirmation emails ({reconfirm_rate}% coverage),
    growing repeat guests (+{loyalty_rate}%), and shifting {deposit_rate}% of bookings to
    non-refundable deposits, you could prevent **{total_saves:,.0f} cancellations** and
    recover approximately **RM {revenue_saved:,.0f}** in revenue.
    """)
