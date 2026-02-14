import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

# 1. Define your initial patient data (blueprint)
seed_data = pd.DataFrame({
    'Patient_ID': range(1, 11),
    'Age': [25, 72, 45, 12, 85, 30, 58, 67, 19, 41],
    'Gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
    'Systolic_BP': [120, 180, 140, 110, 190, 118, 155, 170, 122, 135],
    'Heart_Rate': [72, 115, 88, 95, 120, 70, 94, 110, 72, 82],
    'Symptom': ['Cough', 'Chest Pain', 'Fever', 'Rash', 'Shortness of Breath', 
                'Headache', 'Nausea', 'Dizziness', 'Sore Throat', 'Fatigue']
})

# 2. Tell SDV how to treat your columns
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(seed_data)

# 3. Create and train the synthesizer
synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(seed_data)

# Generate the 5,000 records
synthetic_patients = synthesizer.sample(num_rows=5000)

# Function to apply medical logic to the new data
def label_risk(row):
    if row['Systolic_BP'] > 160 or row['Symptom'] == 'Chest Pain':
        return 'High'
    elif row['Systolic_BP'] > 140 or row['Age'] > 65:
        return 'Medium'
    else:
        return 'Low'

synthetic_patients['Risk_Level'] = synthetic_patients.apply(label_risk, axis=1)

# 4. EXPORT TO CSV
synthetic_patients.to_csv('triage_dataset_5000.csv', index=False)
print("Success! Your 5,000 records are stored in 'triage_dataset_5000.csv'.")