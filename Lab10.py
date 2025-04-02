import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator, ClusteringEvaluator, BinaryClassificationEvaluator

# Initialize Spark Session with increased memory allocation
spark = SparkSession.builder \
    .appName("WomensClothingReviews") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Load Dataset
def load_data():
    file_path = "Womens Clothing E-Commerce Reviews.csv"
    df = pd.read_csv(file_path)
    return df

df = load_data()

st.set_page_config(page_title="Women's Clothing E-Commerce Analysis", layout="wide")
st.title("👗 Women's Clothing E-Commerce Data Analysis")
st.write("### A Big Data Analysis & Machine Learning App using PySpark")

# Convert Pandas DataFrame to PySpark DataFrame
spark_df = spark.createDataFrame(df)

# Initialize session state for storing model results
if "results" not in st.session_state:
    st.session_state.results = {}

# Show raw data
st.subheader("📊 Raw Data Preview")
st.dataframe(df.head())

# Data Cleaning
st.subheader("🛠 Big Data Cleaning & Wrangling in PySpark")

# Show missing values before handling
missing_values = spark_df.select([count(when(col(c).isNull(), 1)).alias(c) for c in spark_df.columns])
st.write("#### Missing Values Before Handling")
st.dataframe(missing_values.toPandas())

# Handling Missing Values
spark_df = spark_df.na.fill({"Title": "No Title", "Review Text": "No Review"})
spark_df = spark_df.dropna()

# Show missing values after handling
missing_values_after = spark_df.select([count(when(col(c).isNull(), 1)).alias(c) for c in spark_df.columns])
st.write("#### Missing Values After Handling")
st.dataframe(missing_values_after.toPandas())

# Limit dataset size to avoid computational issues
spark_df = spark_df.limit(5000)

# Exploratory Data Analysis
st.subheader("📊 Exploratory Data Analysis (EDA)")

# Correlation Heatmap
st.write("#### Correlation Heatmap")
numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
st.pyplot(plt)

# Data Distribution Graphs
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x=df["Rating"], ax=ax, palette="coolwarm")
ax.set_title("Product Rating Distribution")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df["Age"], bins=30, kde=True, color="purple")
ax.set_title("Age Distribution of Reviewers")
st.pyplot(fig)

# Machine Learning with MLlib
st.subheader("🤖 Machine Learning Models")

def prepare_features(df, feature_cols, label_col):
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    df = assembler.transform(df)
    return df.select("features", col(label_col).alias("label"))

# Regression Model
if st.button("Run Regression Model"):
    st.write("Training Linear Regression Model...")
    feature_cols = ["Age", "Positive Feedback Count"]
    reg_data = prepare_features(spark_df, feature_cols, "Rating")
    train, test = reg_data.randomSplit([0.8, 0.2], seed=42)
    reg_model = LinearRegression()
    reg_fit = reg_model.fit(train)
    predictions = reg_fit.transform(test)
    
    # Using Mean Absolute Error (MAE)
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
    mae_score = evaluator.evaluate(predictions)
    st.success(f"✅ Regression Model Trained Successfully! MAE Score: {mae_score:.3f}")
    st.session_state.results["Regression"] = mae_score

# Clustering Model
if st.button("Run Clustering Model"):
    st.write("Training K-Means Clustering Model...")
    cluster_data = prepare_features(spark_df, ["Age", "Positive Feedback Count"], "Rating")
    cluster_data = cluster_data.dropna()
    
    kmeans = KMeans(k=2, seed=1)  
    model = kmeans.fit(cluster_data)  
    predictions = model.transform(cluster_data)
    
    # Using ClusteringEvaluator
    evaluator = ClusteringEvaluator()
    mae_score = evaluator.evaluate(predictions)
    st.success(f"✅ Clustering Model Trained Successfully! MAE Score: {mae_score:.3f}")
    st.session_state.results["Clustering"] = mae_score

# Classification Model
if st.button("Run Classification Model"):
    st.write("Training Logistic Regression Model...")
    class_data = prepare_features(spark_df, ["Age", "Positive Feedback Count"], "Recommended IND")
    train, test = class_data.randomSplit([0.8, 0.2], seed=42)
    clf = LogisticRegression()
    clf_fit = clf.fit(train)
    predictions = clf_fit.transform(test)
    
    # Using ROC-AUC and converting to an error score
    evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    mae_score = 1 - evaluator.evaluate(predictions)  
    st.success(f"✅ Classification Model Trained Successfully! MAE Score: {mae_score:.3f}")
    st.session_state.results["Classification"] = mae_score

    # Confusion Matrix
    conf_matrix = predictions.groupBy("label", "prediction").count().toPandas()
    st.write("### Confusion Matrix")
    st.dataframe(conf_matrix)

# Display Bar Graph Comparing Models (Updates dynamically as models are run)
if st.session_state.results:
    st.subheader("📊 Model Performance Comparison (Mean Absolute Error)")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(st.session_state.results.keys(), st.session_state.results.values(), color=['blue', 'green', 'red'])
    ax.set_xlabel("Model Type")
    ax.set_ylabel("Mean Absolute Error (MAE)")
    ax.set_title("Comparison of Machine Learning Models")
    st.pyplot(fig)

st.write("### 🎉 End of Analysis - Thank You!")
