
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")
st.title("Outlier Visualization")

# Load original and cleaned datasets
df_original = pd.read_csv("household_power_consumption.txt", sep=';')

# Convert to numeric
for col in df_original.columns:
    if col not in ['Date', 'Time']:
        df_original[col] = pd.to_numeric(df_original[col], errors='coerce')

# Drop nulls
df_original.dropna(inplace=True)

# Make a copy for capping
df_capped = df_original.copy()

# Get numeric columns
numeric_cols = df_original.select_dtypes(include=['float64', 'int64']).columns

#Capping
for col in numeric_cols:
    Q1 = df_capped[col].quantile(0.25)
    Q3 = df_capped[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)

# Layout
for col in numeric_cols:
    st.subheader(f"{col} - Before and After Capping")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(y=df_original[col], ax=axs[0], color="salmon")
    axs[0].set_title(f"{col} (Before Capping)")
    sns.boxplot(y=df_capped[col], ax=axs[1], color="lightgreen")
    axs[1].set_title(f"{col} (After Capping)")
    st.pyplot(fig)


st.header("📌 Correlation Heatmap")
st.markdown("This heatmap shows the correlation between all numeric features after capping.")

# Correlation matrix and heatmap
correlation_matrix = df_capped[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax)
ax.set_title("Correlation Heatmap")
st.pyplot(fig)


# R² scores dictionary (example)
model_r2_scores = {
    "Linear Regression": 0.9982,
    "KNN Regression": 0.9983,
    "Decision Tree": 0.9975,
    "Random Forest": 0.9983,
    "Ridge Regression": 0.9982
}

# Convert to DataFrame and multiply by 100
r2_df = pd.DataFrame(list(model_r2_scores.items()), columns=["Model", "R² Value"])
r2_df["R² Value (%)"] = r2_df["R² Value"] * 100

# Display DataFrame
st.title("📈 R² Values of Regression Models (%)")
st.dataframe(r2_df[["Model", "R² Value (%)"]])

# Bar Plot
st.subheader("🔎 Visual Comparison of R² Scores (%)")

fig, ax = plt.subplots()
bars = ax.bar(r2_df["Model"], r2_df["R² Value (%)"], color="skyblue")
ax.set_ylabel("R² Value (%)")
ax.set_ylim(99, 100)
ax.set_yticks(np.arange(99, 100.1, 0.2))  # 99 to 100 with step 0.2
ax.set_title("Model Performance (R² Score %)")
plt.xticks(rotation=45)

# Annotate bars with values
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{height:.2f}%", 
            ha='center', va='bottom')

st.pyplot(fig)


# Model metrics
model_metrics = { 
    "Linear Regression": {"R2": 0.9982, "RMSE": 0.0393, "MAE": 0.0246},
    "KNN Regression": {"R2": 0.9983, "RMSE": 0.0382, "MAE": 0.0214},
    "Decision Tree": {"R2": 0.9975, "RMSE": 0.0460, "MAE": 0.0229},
    "Random Forest": {"R2": 0.9983, "RMSE": 0.0375, "MAE": 0.0201},
    "Ridge Regression": {"R2": 0.9982, "RMSE": 0.0393, "MAE": 0.0246}
}

# Convert to DataFrame
metrics_df = pd.DataFrame(model_metrics).T.reset_index().rename(columns={"index": "Model"})

# Multiply all metrics by 100 and round for neatness
metrics_df["R² (%)"] = (metrics_df["R2"] * 100).round(2)
metrics_df["RMSE (%)"] = (metrics_df["RMSE"] * 100).round(2)
metrics_df["MAE (%)"] = (metrics_df["MAE"] * 100).round(2)

# Final DataFrame to display
display_df = metrics_df[["Model", "R² (%)", "RMSE (%)", "MAE (%)"]]

# Streamlit display
st.subheader("📋 Regression Model Metrics (as %)")
st.dataframe(display_df, use_container_width=True)


