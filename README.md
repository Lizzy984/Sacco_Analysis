SACCO Loan Analytics Dashboard

Overview

The SACCO Loan Analytics Dashboard is a Python-based interactive tool built using Streamlit. It allows users to analyze loan portfolios, assess borrower risk, and explore data-driven insights for decision-making. The dashboard integrates Machine Learning to predict borrower risk categories based on historical loan repayment data.

Live Demo: https://saccoanalysis-lizzy.streamlit.app/

Key Features

· Predictive Modeling: Train a model on filtered data to classify borrowers as Low, Medium, or High Risk
· Portfolio Overview: Key KPIs such as total borrowers, repayment rates, default rates, portfolio value, and average repayment score
· Interactive Visualizations: Charts for risk distribution, portfolio trends, business performance, age-based risk, and repayment status
· Borrower Rankings: Lists top reliable and high-risk borrowers
· Challenges & Solutions: Highlights challenges in loan management and provides potential solutions
· Data Export: Export filtered datasets as CSV for offline analysis

Setup & Installation

1. Clone the repository

```bash
git clone <repository_link>
cd sacco-loan-dashboard
```

2. Create a virtual environment (optional but recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install required packages

```bash
pip install -r requirements.txt
```

4. Run the dashboard

```bash
streamlit run app.py
```

Requirements

Create a requirements.txt file with the following dependencies:

```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
plotly>=5.13.0
scipy>=1.9.0
scikit-learn>=1.2.0
openpyxl>=3.0.0
xlrd>=2.0.0
```

Usage

1. Data Filtering: Use sidebar filters to select data by age group, income level, business type, and loan year
2. Model Training: Train the risk model on filtered data
3. Risk Prediction: Input borrower details to predict risk levels
4. Data Exploration: Explore visualizations and KPIs for insights
5. Data Export: Export filtered data as CSV if needed

Project Structure


sacco-loan-dashboard/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── data/                 # Data directory (optional)
    └── jipange_dataset_cleaned.xlsx
```

Data Schema

The application expects an Excel file with the following columns:

· Borrower_ID: Unique identifier
· Name: Borrower name
· Age: Borrower age (18-35)
· Loan_Amount: Loan amount in KES
· Business_Type: Type of business
· Income_Level: Income category (High/Medium/Low)
· Repayment_Status: Current status (On-time/Late/Default)
· Repayment_Score: Performance score (0.3-0.95)
· Loan_Year: Year loan was issued

Risk Categories

· Low Risk: Reliable borrowers with strong repayment history
· Medium Risk: Requires monitoring and occasional follow-up
· High Risk: Needs immediate attention and possible restructuring

Author

Lizpencer Adhiambo
Zetech University – School Project

Disclaimer

This project is created solely for academic purposes. It is not intended for commercial use or real-world financial decision-making.

Support

For issues or questions:

1. Check that all dependencies are properly installed
2. Ensure your data file matches the expected schema
3. Verify you have sufficient data for model training



Built with ❤️ for SACCOs and financial inclusion initiatives
