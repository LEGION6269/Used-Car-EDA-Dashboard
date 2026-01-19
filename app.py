import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import streamlit as st

df=pd.read_csv(r"C:\Users\Lenovo\OneDrive\Desktop\Dataset_Folder\Cars_cleaned.csv")

#--------------Config---------------

st.set_page_config(
    page_title = 'Used Car EDA Dashboard',
    layout='wide'
)
sns.set_style('whitegrid')

#--------------LOAD DATA-------------

@st.cache_data
def load_data():
    return pd.read_csv('Cars_cleaned.csv')
df = load_data()

#--------------Sidebar--------------

st.sidebar.title('🔍Filters')

location = st.sidebar.multiselect(
    'Select Location',
    df['Location'].unique(),
    default=df['Location'].unique()
)
fuel = st.sidebar.multiselect(
    'Fuel Type',
    df['Fuel_Type'].unique(),
    default=df['Fuel_Type'].unique()
)
transmission = st.sidebar.multiselect(
    'Transmission',
    df['Transmission'].unique(),
    default=df['Transmission'].unique()
)
filtered_df = df[
    (df['Location'].isin(location))&
    (df['Fuel_Type'].isin(fuel))&
    (df['Transmission'].isin(transmission))
]

#---------------Title-----------

st.title('🚗💨Used Car Price Analysis Dashboard')
st.markdown('### End-to-End Exploratory Data Analysis (EDA) using streamlit')

#---------------Introduction---------

st.title('📝Project Introduction')
st.markdown('''The used car market in India has grown rapidly and now exceeds the new car market.
            This dashboard performs ** end-to-end exploratory data analysis ** to identify key factors affecting used car practices''')

#---------------Data Overview--------

st.header('🎯Dataset Overview')

col1, col2, col3 = st.columns(3)
col1.metric('Total Records', df.shape[0])
col2.metric('Total Feautres', df.shape[1])
col3.metric('Locations',df['Location'].nunique())

st.dataframe(filtered_df.head())

#---------------Data Modelling ------------

st.header('🧹Data modelling & Clean memory')
st.write('✅ Handled missiing values')
st.write('✅ Converted mileage, engine & power to numeric')
st.write('✅ Removed outliers where required')

#--------------Numerical Analysis-------------

st.header("📉 Numerical Feature Analysis")

num_cols = ["Kilometers_Driven", "Mileage_value", "Engine_value", "Power_value", "Price"]

selected_num = st.selectbox(
    "Select Numerical Feature",
    num_cols
)

fig1, ax1 = plt.subplots()
sns.histplot(filtered_df[selected_num], kde=True, ax=ax1)
st.pyplot(fig1)

#--------------Categorical Analysis----------------

st.header("📊 Categorical Feature Analysis")

cat_cols = ["Fuel_Type", "Transmission", "Owner", "Seats"]

selected_cat = st.selectbox(
    "Select Categorical Feature",
    cat_cols
)

fig2, ax2 = plt.subplots()
sns.countplot(
    data=filtered_df,
    x=selected_cat,
    order=filtered_df[selected_cat].value_counts().index,
    ax=ax2
)
plt.xticks(rotation=45)
st.pyplot(fig2)

# ---------------- CORRELATION ----------------

st.header("🔗 Correlation with Price")

corr = filtered_df[num_cols].corr()

fig3, ax3 = plt.subplots(figsize=(8, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax3)
st.pyplot(fig3)

#----------------

# ---------------- KEY INSIGHTS ----------------

st.header("💡 Key Business Insights")

st.markdown("""
- **Kilometers Driven** has strong negative correlation (≈ -0.63) with price  
- **Engine & Power** show strong positive relationship with price  
- **Automatic cars** are priced higher than manual  
- **Diesel cars** retain better resale value  
- Location impacts pricing trends
""")

# ---------------- CONCLUSION ----------------

st.header("🚩Conclusion")

st.markdown("""
This project demonstrates a **complete data analyst workflow**:
- Business understanding
- Data cleaning & modeling
- Automated EDA
- Interactive visualizations
- Insight-driven conclusions

This dashboard can help **dealers, buyers, and pricing teams**
make informed, data-driven decisions.
""")
