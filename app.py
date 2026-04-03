import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Titanic Dashboard", layout="wide")

# ---------------- TITLE ----------------
st.title("🚢 Titanic Survival Prediction Dashboard")
st.markdown("### Interactive ML App with Prediction, Visuals & Insights")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("Titanic-Dataset.csv")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Visualizations", "📈 Model Insights"])

# =========================================================
# 🔮 TAB 1: PREDICTION
# =========================================================
with tab1:
    st.subheader("Enter Passenger Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        Pclass = st.selectbox("Passenger Class", [1, 2, 3])
        Sex = st.selectbox("Gender", ["Male", "Female"])

    with col2:
        Age = st.slider("Age", 0, 80, 25)
        Fare = st.slider("Fare", 0, 500, 50)

    with col3:
        SibSp = st.slider("Siblings/Spouses", 0, 5, 0)
        Parch = st.slider("Parents/Children", 0, 5, 0)

    Embarked = st.selectbox("Port of Embarkation", ["Q", "S", "C"])

    # Convert inputs
    Sex_val = 1 if Sex == "Female" else 0
    Embarked_Q = 1 if Embarked == "Q" else 0
    Embarked_S = 1 if Embarked == "S" else 0

    FamilySize = SibSp + Parch
    IsAlone = 1 if FamilySize == 0 else 0

    input_data = np.array([[Pclass, Sex_val, Age, SibSp, Parch, Fare,
                            Embarked_Q, Embarked_S, FamilySize, IsAlone]])

    if st.button("Predict Survival"):
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.success("✅ The passenger is likely to SURVIVE")
        else:
            st.error("❌ The passenger is NOT likely to survive")

# =========================================================
# 📊 TAB 2: VISUALIZATIONS
# =========================================================
with tab2:
    st.subheader("Data Visualizations")

    col1, col2 = st.columns(2)

    # Survival count
    with col1:
        fig1, ax1 = plt.subplots()
        sns.countplot(x='Survived', data=df, ax=ax1)
        ax1.set_title("Survival Count")
        st.pyplot(fig1)

    # Survival by gender
    with col2:
        fig2, ax2 = plt.subplots()
        sns.countplot(x='Sex', hue='Survived', data=df, ax=ax2)
        ax2.set_title("Survival by Gender")
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    # Survival by class
    with col3:
        fig3, ax3 = plt.subplots()
        sns.countplot(x='Pclass', hue='Survived', data=df, ax=ax3)
        ax3.set_title("Survival by Class")
        st.pyplot(fig3)

    # Age distribution
    with col4:
        fig4, ax4 = plt.subplots()
        sns.histplot(df['Age'].dropna(), bins=30, kde=True, ax=ax4)
        ax4.set_title("Age Distribution")
        st.pyplot(fig4)

    col5, col6 = st.columns(2)

    # Fare distribution
    with col5:
        fig5, ax5 = plt.subplots()
        sns.histplot(df['Fare'], bins=30, kde=True, ax=ax5)
        ax5.set_title("Fare Distribution")
        st.pyplot(fig5)

    # Correlation heatmap
    with col6:
        fig6, ax6 = plt.subplots(figsize=(6,4))
        sns.heatmap(df.corr(numeric_only=True), annot=True, ax=ax6)
        ax6.set_title("Correlation Heatmap")
        st.pyplot(fig6)

# =========================================================
# 📈 TAB 3: MODEL INSIGHTS
# =========================================================
with tab3:
    st.subheader("Model Performance")

    # Accuracy
    st.metric("Model Accuracy", "0.82")

    # Preprocess for confusion matrix
    X = df.drop(columns=['Survived', 'Name', 'Ticket', 'PassengerId'], errors='ignore')
    y = df['Survived']

    X['Age'].fillna(X['Age'].mean(), inplace=True)
    X['Embarked'].fillna(X['Embarked'].mode()[0], inplace=True)
    X.drop(columns=['Cabin'], inplace=True, errors='ignore')

    X['Sex'] = X['Sex'].map({'male':0, 'female':1})
    X = pd.get_dummies(X, columns=['Embarked'], drop_first=True)

    X['FamilySize'] = X['SibSp'] + X['Parch']
    X['IsAlone'] = (X['FamilySize'] == 0).astype(int)

    # Predictions
    preds = model.predict(X)

    cm = confusion_matrix(y, preds)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # Insights
    st.markdown("### 🔍 Insights")
    st.write("""
    - Gender and passenger class strongly influence survival  
    - Females had higher survival chances  
    - Higher class passengers were more likely to survive  
    - Model shows good predictive performance with 82% accuracy  
    """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")