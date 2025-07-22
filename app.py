import streamlit as st
import pandas as pd
import pickle

pipe = pickle.load(open('best_model_pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.set_page_config(page_title="Employee Salary Prediction", page_icon="💼", layout="centered")

st.title("💼 Earnlytics : Smart Salary Predictor")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features")

age = st.sidebar.slider('Age', 18, 65, 30)

education = st.sidebar.selectbox('Education',df['education'].unique())

workclass = st.sidebar.selectbox('Workclass',df['workclass'].unique())

occupation =st.sidebar.selectbox('Occupation',df['occupation'].unique())

marital_status = st.sidebar.selectbox('Marital Status', df['marital_status'].unique())

relationship = st.sidebar.selectbox('Relationship',df['relationship'].unique())

gender = st.sidebar.radio('Select Gender',df['gender'].unique())

hours_per_week = st.sidebar.slider('Hours per Week', 
                                   int(df['hours_per_week'].min()), 
                                   int(df['hours_per_week'].max()), 
                                   step=1)

race = st.sidebar.selectbox('Race', df['race'].unique())

country_group = st.sidebar.selectbox('Native Country', df['country_group'].unique())

has_capital_gain = st.sidebar.selectbox('Capital Gain', df['has_capital_gain'].unique())

has_capital_loss = st.sidebar.selectbox('Capital Loss', df['has_capital_loss'].unique())

# Build input DataFrame (⚠️ must match preprocessing of your training data)
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'workclass' : [workclass],
    'occupation': [occupation],
    'marital_status' : [marital_status],
    'relationship' : [relationship],
    'gender' : [gender],
    'hours_per_week': [hours_per_week],
    'race' : [race],
    'country_group' : [country_group],
    'has_capital_gain' : [has_capital_gain],
    'has_capital_loss' : [has_capital_loss]
    
})

st.write("### 🔎 Input Data")
st.write(input_df)

if st.button("Predict Salary"):
    prediction = pipe.predict(input_df)
    salary_class = prediction[0]

    # Map numeric class to human-readable label
    salary_map = {0: "<=50K", 1: ">50K"}
    readable_class = salary_map.get(salary_class, "Unknown")

    st.success(f"✅ Predicted Salary: **{readable_class}**")
    if readable_class == ">50K":
        st.info("📈 This person is predicted to earn more than $50,000 annually.")
    else:
        st.warning("📉 This person is predicted to earn $50,000 or less annually.")

        
        
# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())
    batch_preds = pipe.predict(batch_data)
    # Map numeric predictions to actual labels
    label_map = {0: "<=50K", 1: ">50K"}
    batch_data['PredictedClass'] = batch_preds
    batch_data['PredictedClass'] = batch_data['PredictedClass'].map(label_map)

    # Display predictions
    st.write("✅ Predictions:")
    st.write(batch_data.head())

    # Download updated CSV
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
