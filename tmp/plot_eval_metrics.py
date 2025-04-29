import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# ------------------------------------------------------------------
# 1) Copy your CSV lines into a string (or load from a file).
#    Below, I'm putting your CSV table literally in triple-quotes.
# ------------------------------------------------------------------
csv_data = """FieldName,TP,FP,TN,FN,Accuracy,Precision,Recall,F1
Alcohol_Consumption,39,1,0,10,0.78,0.975,0.7959183673469388,0.8764044943820225
Anatomic_Site_of_Lesion,50,0,0,0,1.0,1.0,1.0,1.0
Charlson_Comorbidity_Score,32,17,0,0,0.6530612244897959,0.6530612244897959,1.0,0.7901234567901235
Clinical_Assessments_Radiological_Lesions,43,0,0,6,0.8775510204081632,1.0,0.8775510204081632,0.9347826086956522
Clinical_Assessments_SUV_from_PET_scans,4,0,0,45,0.08163265306122448,1.0,0.08163265306122448,0.15094339622641506
Clinical_TNM,36,2,0,12,0.72,0.9473684210526315,0.75,0.8372093023255814
EBER_Status,3,0,0,47,0.06,1.0,0.06,0.11320754716981131
ECOG_Performance_Status,44,0,0,5,0.8979591836734694,1.0,0.8979591836734694,0.9462365591397849
Follow_Up_Plans,45,0,0,5,0.9,1.0,0.9,0.9473684210526316
HPV_Status,38,0,0,12,0.76,1.0,0.76,0.8636363636363636
Immunohistochemical_profile,40,0,0,10,0.8,1.0,0.8,0.888888888888889
Karnofsky_Performance_Status,43,1,0,5,0.8775510204081632,0.9772727272727273,0.8958333333333334,0.9347826086956522
Lymph_Node_Status_Extranodal_Extension,32,1,0,17,0.64,0.9696969696969697,0.6530612244897959,0.7804878048780488
Lymph_Node_Status_Number_of_Positve_Lymph_Nodes,45,2,0,3,0.9,0.9574468085106383,0.9375,0.9473684210526315
Lymph_Node_Status_Presence_Absence,48,0,0,2,0.96,1.0,0.96,0.9795918367346939
Lymphovascular_Invasion_Status,27,1,0,22,0.54,0.9642857142857143,0.5510204081632653,0.7012987012987012
Pack_Years,39,1,0,10,0.78,0.975,0.7959183673469388,0.8764044943820225
Pathological_TNM,21,0,1,28,0.44,1.0,0.42857142857142855,0.6
Pathology_Details,49,1,0,0,0.98,0.98,1.0,0.98989898989899
Patient_History_Status_Previous_Treatments,45,0,0,4,0.9183673469387755,1.0,0.9183673469387755,0.9574468085106383
Patient_History_Status_Prior_Conditions,42,1,0,7,0.84,0.9767441860465116,0.8571428571428571,0.9130434782608695
Patient_Symptoms_at_Presentation,48,0,0,2,0.96,1.0,0.96,0.9795918367346939
Perineural_Invasion_Status,22,3,0,25,0.44,0.88,0.46808510638297873,0.6111111111111112
Primary_Tumor_Size,43,1,0,6,0.86,0.9772727272727273,0.8775510204081632,0.9247311827956989
Resection_Margins,25,0,0,25,0.5,1.0,0.5,0.6666666666666666
Sex,43,2,1,3,0.8979591836734694,0.9555555555555556,0.9347826086956522,0.945054945054945
Smoking_History,45,0,0,5,0.9,1.0,0.9,0.9473684210526316
Treatment_Recommendations,45,0,0,5,0.9,1.0,0.9,0.9473684210526316
Tumor_Type_Differentiation,41,3,0,6,0.82,0.9318181818181818,0.8723404255319149,0.9010989010989012
p16_Status,38,0,0,12,0.76,1.0,0.76,0.8636363636363636
"""

# ------------------------------------------------------------------
# 2) Read the CSV data into a DataFrame
# ------------------------------------------------------------------
df = pd.read_csv(StringIO(csv_data))

# ------------------------------------------------------------------
# 3) Melt the DataFrame so we can easily plot grouped bars
#    We'll plot "Accuracy", "Precision", "Recall", "F1".
# ------------------------------------------------------------------
metrics = ["Accuracy", "Precision", "Recall", "F1"]
df_melt = df.melt(
    id_vars="FieldName",
    value_vars=metrics,
    var_name="Metric",
    value_name="Score"
)
print(f"\n--- Melted DataFrame ---")
print(df_melt.head())
print(df_melt)

# ------------------------------------------------------------------
# 4) Create the grouped bar plot
# ------------------------------------------------------------------
plt.figure(figsize=(10, 12))
sns.barplot(
    data=df_melt,
    x="Score", 
    y="FieldName",
    hue="Metric"
)

plt.title("Performance Metrics by Field Name", fontsize=14)
plt.xlabel("Metric Score")
plt.ylabel("Field Name")
plt.xlim(0, 1.05)  # because some metrics are up to 1.0
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 5) Save as a table in CSV format
# ------------------------------------------------------------------
df_melt.to_csv("/Data/Yujing/HNC_OutcomePred/Reports_Agents/tmp/melted_metrics.csv", index=False)
print("\n--- Melted DataFrame Saved as melted_metrics.csv ---")
# ------------------------------------------------------------------