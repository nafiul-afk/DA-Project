import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
TEAL = '#0f766e'
DARK = '#134e4a'

df = pd.read_csv('cleaned_data.csv')

# 1. Readmission rate by age band
fig, ax = plt.subplots(figsize=(7, 4.2))
rate = df.groupby('age')['readmit_30'].mean().sort_index() * 100
rate.plot(kind='bar', ax=ax, color=TEAL)
ax.set_ylabel('30-day readmission rate (%)')
ax.set_xlabel('Age band')
ax.set_title('Readmission Rate by Age Group')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('fig1_age_readmit.png')
plt.close()

# 2. Readmission rate by number of prior inpatient visits (H1)
fig, ax = plt.subplots(figsize=(7, 4.2))
df['inpatient_bucket'] = df['number_inpatient'].clip(upper=5)
rate2 = df.groupby('inpatient_bucket')['readmit_30'].mean() * 100
rate2.plot(kind='bar', ax=ax, color=TEAL)
ax.set_ylabel('30-day readmission rate (%)')
ax.set_xlabel('Prior inpatient visits (5 = 5 or more)')
ax.set_title('H1: Prior Inpatient Visits vs. Readmission Risk')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('fig2_priorvisits_readmit.png')
plt.close()

# 3. Length of stay distribution split by readmitted status (H2)
fig, ax = plt.subplots(figsize=(7, 4.2))
sns.kdeplot(data=df, x='time_in_hospital', hue='readmit_30', fill=True,
            common_norm=False, palette=[DARK, TEAL], ax=ax)
ax.set_xlabel('Time in hospital (days)')
ax.set_title('H2: Length of Stay by Readmission Status')
plt.tight_layout()
plt.savefig('fig3_los_readmit.png')
plt.close()

# 4. Insulin change vs readmission (H3)
fig, ax = plt.subplots(figsize=(7, 4.2))
rate3 = df.groupby('insulin')['readmit_30'].mean().sort_values(ascending=False) * 100
rate3.plot(kind='bar', ax=ax, color=TEAL)
ax.set_ylabel('30-day readmission rate (%)')
ax.set_xlabel('Insulin dosage change')
ax.set_title('H3: Insulin Change vs. Readmission Risk')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('fig4_insulin_readmit.png')
plt.close()

# 5. Readmission rate by primary diagnosis group
fig, ax = plt.subplots(figsize=(7.5, 4.5))
rate4 = df.groupby('diag_1_group')['readmit_30'].mean().sort_values(ascending=False) * 100
rate4.plot(kind='barh', ax=ax, color=TEAL)
ax.set_xlabel('30-day readmission rate (%)')
ax.set_title('Readmission Rate by Primary Diagnosis Category')
plt.tight_layout()
plt.savefig('fig5_diag_readmit.png')
plt.close()

# 6. Target class imbalance
fig, ax = plt.subplots(figsize=(5, 4.2))
df['readmit_30'].value_counts().rename({0: 'No <30d readmit', 1: '<30d readmit'}).plot(
    kind='bar', ax=ax, color=[DARK, TEAL])
ax.set_ylabel('Number of patients')
ax.set_title('Class Balance: 30-Day Readmission')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('fig6_class_balance.png')
plt.close()

# Print summary stats used in report text
print("Readmit rate by age:\n", rate.round(2))
print("\nReadmit rate by prior inpatient visits:\n", rate2.round(2))
print("\nReadmit rate by insulin change:\n", rate3.round(2))
print("\nReadmit rate by diagnosis group:\n", rate4.round(2))
print("\nMean LOS - readmitted:", df[df.readmit_30==1].time_in_hospital.mean().round(2),
      " vs not:", df[df.readmit_30==0].time_in_hospital.mean().round(2))
print("\nAll figures saved.")