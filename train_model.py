import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# 1. LOAD YOUR HIGH-QUALITY DATA
df = pd.read_csv('triage_v3_90plus.csv')

# 2. ENCODE CATEGORICAL DATA
# AI models only understand numbers, so we turn "Chest Pain" into 0, 1, 2...
le_gender = LabelEncoder()
le_symptoms = LabelEncoder()    
le_pre = LabelEncoder()

df['Gender'] = le_gender.fit_transform(df['Gender'])
df['Symptoms'] = le_symptoms.fit_transform(df['Symptoms'])
df['Pre_Existing'] = le_pre.fit_transform(df['Pre_Existing'])

# 3. DEFINE X (Features) and Y (Target)
X = df[['Age', 'Gender', 'Systolic_BP', 'Heart_Rate', 'Symptoms', 'Pre_Existing']]
y = df['Risk_Level']

# 4. SPLIT DATA (80% to learn, 20% to test its own knowledge)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. TRAIN THE RANDOM FOREST
print("Training the AI Brain... please wait.")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. EVALUATE PERFORMANCE
y_pred = model.predict(X_test)
print("\n--- AI PERFORMANCE REPORT ---")
print(classification_report(y_test, y_pred))

# 7. SAVE THE BRAIN AND THE ENCODERS
# You need these files to run your website later!
joblib.dump(model, 'triage_model.pkl')
joblib.dump(le_gender, 'le_gender.pkl')
joblib.dump(le_symptoms, 'le_symptoms.pkl')
joblib.dump(le_pre, 'le_pre.pkl')

print("\nSUCCESS: AI Model saved as 'triage_model.pkl'")