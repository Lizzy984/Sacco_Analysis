import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="LOAN ANALYTICS DASHBOARD", layout="wide")

def load_data():
    try:
        df = pd.read_excel("C:/Users/Liz/OneDrive/Desktop/JIPANGE_SACCO_ANALYSIS/jipange_dataset_cleaned.xlsx")
        return df
    except Exception as e:
        st.error(f"File load failed: {e}")
        np.random.seed(42)
        data = {
            'Borrower_ID': range(1, 101),
            'Name': [f'Borrower_{i}' for i in range(1, 101)],
            'Age': np.random.randint(18, 36, 100),
            'Loan_Amount': np.random.randint(5000, 50000, 100),
            'Business_Type': np.random.choice(['Salon', 'Farming', 'Retail', 'Boda Boda', 'Other'], 100),
            'Income_Level': np.random.choice(['High', 'Medium', 'Low'], 100),
            'Repayment_Status': np.random.choice(['On-time', 'Late', 'Default'], 100, p=[0.6, 0.25, 0.15]),
            'Repayment_Score': np.random.uniform(0.3, 0.95, 100),
            'Loan_Year': np.random.choice([2021, 2022, 2023, 2024], 100)
        }
        return pd.DataFrame(data)

def process_data(df):
    df_clean = df.copy()
    
    def risk_classifier(score, status):
        if status == 'Default': return 'High Risk'
        if status == 'Late': return 'Medium Risk' if score > 0.5 else 'High Risk'
        return 'Low Risk' if score > 0.7 else 'Medium Risk'
    
    df_clean['Risk_Category'] = df_clean.apply(lambda x: risk_classifier(x['Repayment_Score'], x['Repayment_Status']), axis=1)
    df_clean['Age_Group'] = pd.cut(df_clean['Age'], [17, 25, 30, 36], labels=['18-25', '26-30', '31-35'])
    return df_clean

def train_model(df):
    features = ['Age', 'Loan_Amount', 'Income_Level', 'Business_Type', 'Loan_Year']
    target = 'Risk_Category'
    
    if df[target].nunique() < 2:
        raise ValueError("Need multiple risk categories")
    
    le_income = LabelEncoder()
    le_business = LabelEncoder()
    le_risk = LabelEncoder()
    
    X = df[features].copy()
    X['Income_Level_Enc'] = le_income.fit_transform(X['Income_Level'])
    X['Business_Type_Enc'] = le_business.fit_transform(X['Business_Type'])
    X = X[['Age', 'Loan_Amount', 'Income_Level_Enc', 'Business_Type_Enc', 'Loan_Year']]
    
    y = le_risk.fit_transform(df[target])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, le_income, le_business, le_risk, accuracy

if 'model_tracker' not in st.session_state:
    st.session_state.model_tracker = []
if 'current_model' not in st.session_state:
    st.session_state.current_model = None

df_raw = load_data()
df_processed = process_data(df_raw)

st.title("LOAN ANALYTICS DASHBOARD")

with st.sidebar.expander("Help"):
    st.write("""
    How to use:
    1. Filter data using controls
    2. Train model when ready
    3. Check risk for new applicants
    4. View charts and rankings
    """)

st.sidebar.header("Filters")
age_filter = st.sidebar.multiselect("Age Groups", ['18-25', '26-30', '31-35'], default=['18-25', '26-30', '31-35'])
income_filter = st.sidebar.multiselect("Income Levels", ['High', 'Medium', 'Low'], default=['High', 'Medium', 'Low'])
business_filter = st.sidebar.multiselect("Business Types", df_processed['Business_Type'].unique().tolist(), default=df_processed['Business_Type'].unique().tolist())
year_filter = st.sidebar.multiselect("Loan Years", sorted(df_processed['Loan_Year'].unique()), default=sorted(df_processed['Loan_Year'].unique()))

df_filtered = df_processed[
    (df_processed['Age_Group'].isin(age_filter)) &
    (df_processed['Income_Level'].isin(income_filter)) &
    (df_processed['Business_Type'].isin(business_filter)) &
    (df_processed['Loan_Year'].isin(year_filter))
]

st.sidebar.header("Model")

if st.session_state.current_model:
    st.sidebar.info(f"Model accuracy: {st.session_state.current_model['accuracy']:.1%}")

if st.sidebar.button("Train Model"):
    if len(df_filtered) < 10:
        st.sidebar.error("Not enough data")
    elif df_filtered['Risk_Category'].nunique() < 2:
        st.sidebar.error("Need multiple risk categories")
    else:
        with st.spinner("Training model..."):
            try:
                model, le_inc, le_bus, le_rsk, accuracy = train_model(df_filtered)
                st.session_state.current_model = {
                    'model': model, 
                    'encoders': (le_inc, le_bus, le_rsk),
                    'accuracy': accuracy, 
                    'trained_at': datetime.now(),
                    'training_samples': len(df_filtered)
                }
                st.session_state.model_tracker.append({
                    'timestamp': datetime.now(), 
                    'accuracy': accuracy,
                    'samples': len(df_filtered)
                })
                st.sidebar.success(f"Model trained: {accuracy:.1%}")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Training error: {str(e)}")

if st.session_state.current_model:
    st.sidebar.header("Risk Check")
    
    with st.sidebar.form("predict"):
        p_age = st.slider("Age", 18, 35, 25)
        p_loan = st.number_input("Loan Amount", 5000, 100000, 30000)
        p_income = st.selectbox("Income", ["Low", "Medium", "High"])
        p_business = st.selectbox("Business", ["Salon", "Farming", "Retail", "Boda Boda", "Other"])
        
        if st.form_submit_button("Check Risk"):
            model_data = st.session_state.current_model
            le_inc, le_bus, le_rsk = model_data['encoders']
            
            input_data = pd.DataFrame([{
                'Age': p_age, 
                'Loan_Amount': p_loan,
                'Income_Level_Enc': le_inc.transform([p_income])[0],
                'Business_Type_Enc': le_bus.transform([p_business])[0],
                'Loan_Year': 2024
            }])
            
            prediction = model_data['model'].predict(input_data)[0]
            risk_level = le_rsk.inverse_transform([prediction])[0]
            
            if risk_level == "Low Risk":
                st.sidebar.success(f"Risk: {risk_level}")
            elif risk_level == "Medium Risk":
                st.sidebar.warning(f"Risk: {risk_level}")
            else:
                st.sidebar.error(f"Risk: {risk_level}")

total_borrowers = len(df_filtered)
on_time_rate = (df_filtered['Repayment_Status'] == 'On-time').mean() * 100
default_rate = (df_filtered['Repayment_Status'] == 'Default').mean() * 100
portfolio_value = df_filtered['Loan_Amount'].sum()
avg_score = df_filtered['Repayment_Score'].mean()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Borrowers", total_borrowers)
col2.metric("On-time Rate", f"{on_time_rate:.1f}%")
col3.metric("Default Rate", f"{default_rate:.1f}%")
col4.metric("Portfolio Value", f"KES {portfolio_value:,.0f}")
col5.metric("Avg Score", f"{avg_score:.2f}")

st.subheader("Analytics")

col1, col2 = st.columns(2)

with col1:
    risk_data = df_filtered['Risk_Category'].value_counts()
    fig_risk = px.pie(values=risk_data.values, names=risk_data.index, title="Risk Distribution")
    st.plotly_chart(fig_risk, use_container_width=True)
    
    with st.expander("Risk Distribution Analysis"):
        st.write("""
        **Summary Analysis:**
        - Shows the proportion of borrowers in each risk category
        - Low Risk: Reliable borrowers with strong repayment history
        - Medium Risk: Requires monitoring and occasional follow-up
        - High Risk: Needs immediate attention and possible loan restructuring
        - **Key Insight:** Monitor medium-risk borrowers closely to prevent escalation to high-risk
        """)

with col2:
    yearly_data = df_filtered.groupby('Loan_Year').agg({'Loan_Amount': 'sum'}).reset_index()
    fig_trend = px.line(yearly_data, x='Loan_Year', y='Loan_Amount', title="Portfolio Trend")
    st.plotly_chart(fig_trend, use_container_width=True)
    
    with st.expander("Portfolio Trend Analysis"):
        st.write("""
        **Summary Analysis:**
        - Tracks the total loan portfolio value over time
        - Upward trend indicates growing lending activity
        - Downward trend may signal reduced lending or increased defaults
        - **Key Insight:** Use this to forecast future portfolio growth and capital requirements
        """)

col3, col4 = st.columns(2)

with col3:
    biz_data = df_filtered.groupby('Business_Type')['Repayment_Score'].mean().reset_index()
    fig_biz = px.bar(biz_data, x='Business_Type', y='Repayment_Score', title="Business Performance")
    st.plotly_chart(fig_biz, use_container_width=True)
    
    with st.expander("Business Performance Analysis"):
        st.write("""
        **Summary Analysis:**
        - Compares average repayment scores across different business types
        - Higher scores indicate more reliable business sectors
        - Lower scores suggest higher risk business categories
        - **Key Insight:** Use this to adjust lending policies for different business types
        """)

with col4:
    age_risk = pd.crosstab(df_filtered['Age_Group'], df_filtered['Risk_Category'])
    fig_age = px.bar(age_risk, barmode='group', title="Age vs Risk")
    st.plotly_chart(fig_age, use_container_width=True)
    
    with st.expander("Age vs Risk Analysis"):
        st.write("""
        **Summary Analysis:**
        - Shows how risk distribution varies across age groups
        - Younger borrowers may show different risk patterns
        - Older borrowers might demonstrate more stability
        - **Key Insight:** Tailor financial education and support based on age-specific risk patterns
        """)

col5, col6 = st.columns(2)

with col5:
    fig_loan_dist = px.histogram(df_filtered, x='Loan_Amount', title="Loan Amount Distribution", nbins=15)
    st.plotly_chart(fig_loan_dist, use_container_width=True)
    
    with st.expander("Loan Amount Analysis"):
        st.write("""
        **Summary Analysis:**
        - Shows the distribution of loan sizes in the portfolio
        - Identifies most common loan amount ranges
        - Helps detect concentration risk in specific loan sizes
        - **Key Insight:** Optimize loan product offerings based on popular amount ranges
        """)

with col6:
    status_data = df_filtered['Repayment_Status'].value_counts()
    fig_status = px.bar(x=status_data.index, y=status_data.values, title="Repayment Status")
    st.plotly_chart(fig_status, use_container_width=True)
    
    with st.expander("Repayment Status Analysis"):
        st.write("""
        **Summary Analysis:**
        - Shows count of borrowers by their current repayment status
        - On-time: Meeting payment obligations
        - Late: Behind schedule but still paying
        - Default: Significant delinquency or non-payment
        - **Key Insight:** Focus collection efforts on late payers to prevent defaults
        """)

if st.session_state.current_model:
    st.subheader("Model Insights")
    
    model_data = st.session_state.current_model
    model = model_data['model']
    
    col1, col2 = st.columns(2)
    
    with col1:
        feature_names = ['Age', 'Loan Amount', 'Income Level', 'Business Type', 'Loan Year']
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig_importance = px.bar(importance_df, x='Importance', y='Feature', title="Feature Importance", orientation='h')
        st.plotly_chart(fig_importance, use_container_width=True)
        
        with st.expander("Feature Importance Analysis"):
            st.write("""
            **Summary Analysis:**
            - Shows which factors most influence risk predictions
            - Higher importance = stronger impact on risk classification
            - Income Level and Loan Amount typically strongest predictors
            - **Key Insight:** Focus on high-importance features for borrower assessment
            """)

    with col2:
        st.write("Model Details")
        st.metric("Accuracy", f"{model_data['accuracy']:.1%}")
        st.metric("Trained", model_data['trained_at'].strftime('%Y-%m-%d'))
        st.metric("Samples", model_data['training_samples'])

st.subheader("Borrower Rankings")

col1, col2 = st.columns(2)

with col1:
    st.write("Top 10 Reliable")
    reliable = df_filtered.nlargest(10, 'Repayment_Score')
    st.dataframe(reliable[['Name', 'Business_Type', 'Repayment_Score', 'Loan_Amount']], use_container_width=True)

with col2:
    st.write("Top 10 High Risk")
    risky = df_filtered[df_filtered['Risk_Category'] == 'High Risk'].nsmallest(10, 'Repayment_Score')
    st.dataframe(risky[['Name', 'Business_Type', 'Repayment_Score', 'Loan_Amount']], use_container_width=True)

st.subheader("Challenges")
st.write("Unclear repayment trends")
st.write("High default rates")
st.write("Manual risk assessment")
st.write("Difficulty identifying reliable borrowers")

st.subheader("Solutions")
st.write("Predictive risk scoring")
st.write("Automated assessment")
st.write("Data-driven decisions")
st.write("Performance monitoring")

with st.expander("Export Data"):
    st.dataframe(df_filtered, use_container_width=True)
    csv = df_filtered.to_csv(index=False)
    st.download_button("Download CSV", data=csv, file_name="sacco_data.csv")