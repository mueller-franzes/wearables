from pathlib import Path 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

# -------------- Settings -----------
plt.rcParams['font.size'] = 11
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.axisbelow'] = True
fontdict = {'fontweight':'bold'}
path_out = Path('results/')
path_out.mkdir(exist_ok=True)

# ------------- Read Data ------------
df_lower_outcome = pd.read_excel('data/Lower_Extremity_Patient_Outcomes.xlsx') 
df_upper_outcome = pd.read_excel('data/Upper_Extremity_Patient_Outcomes.xlsx') 
df_lower_steps = pd.read_excel('data/Lower_Extremity_Step_Count_Data.xlsx').set_index('Days').T 
df_upper_steps = pd.read_excel('data/Upper_Extremity_Step_Count_Data.xlsx').set_index('Days').T

print()
assert len(df_lower_outcome) == len(df_lower_steps)
assert len(df_upper_outcome) == len(df_upper_steps)

df_outcome = pd.concat([df_lower_outcome, df_upper_outcome])
df_steps = pd.concat([df_lower_steps, df_upper_steps])
assert len(df_outcome) == len(df_steps)

# patient 128 appears in both - upper and lower - remove
df_outcome = df_outcome[~df_outcome['Patient_ID'].duplicated(keep='first')].reset_index(drop=True)
df_steps = df_steps[~df_steps.index.duplicated(keep='first')]

# Normalize Steps (100%)
df_steps_pre = df_steps[df_steps.columns[df_steps.columns<0]]
lower = df_steps.min(axis=1).values[:, None]
upper = df_steps_pre.quantile(0.975, axis=1).values[:, None]
df_steps = df_steps.clip(lower=lower, upper=upper, axis=1)  
df_steps  = (df_steps-lower)/(upper-lower)*100

# Long format 
df_steps_long = df_steps.reset_index().melt(id_vars='index', var_name='day', value_name='steps')
df_steps_long.rename(columns={'index': 'Patient_ID'}, inplace=True)

# Add outcome 
df = pd.merge(df_steps_long, df_outcome, how='inner', on='Patient_ID')

# Add statistics 
df['PROM_12_Q1'] = (df['PROM_12'] <= df['PROM_12'].quantile(0.25)).astype(int)


#------------------------------------ Plot Extremity ---------------------------------------
# Define titles and subsets of data
extremity_data = [
    ("All Extremity", df, 0),
    ("Lower Extremity", df[df['Patient_ID'].isin(df_lower_outcome.index)], 1),
    ("Upper Extremity", df[df['Patient_ID'].isin(df_upper_outcome.index)], 2)
]

# Loop through each extremity
fig, ax = plt.subplots(3, 2, figsize=(3*8, 2*6))
for i, (title, data, row) in enumerate(extremity_data):
    # Plot Mean ± 95 CI
    axis = ax[row, 0]
    sns.lineplot(data=data, x='day', y='steps', ax=axis)
    axis.axvline(x=0, linestyle='--', color='gray')
    axis.set_title(f"{title}: Mean ± 95 CI", fontdict=fontdict)
    axis.set_xlabel('Time [days]', fontdict=fontdict)
    axis.set_ylabel('Steps [%]', fontdict=fontdict)

    # Plot Individual Patient Data
    axis = ax[row, 1]
    sns.scatterplot(data=data, x="day", y="steps", hue='Patient_ID', legend=False, ax=axis)
    axis.axvline(x=0, linestyle='--', color='gray')
    axis.set_title(f"{title}: Individual", fontdict=fontdict)
    axis.set_xlabel('Time [days]', fontdict=fontdict)
    axis.set_ylabel('Steps [%]', fontdict=fontdict)

fig.tight_layout()
fig.savefig(path_out/'steps.png', dpi=300)




#------------------------------------ Plot Hue ---------------------------------------
# Define titles and subsets of data
hue_data = [
    ("Return to work", "Return_to_work"),
    ("Treatment", "Treatment"),
     ("PROM 12 Q1", "PROM_12_Q1")
]

# Loop through each extremity
fig, ax = plt.subplots(3, 1, figsize=(1*8, 3*4))
for row, (title, hue) in enumerate(hue_data):
    # Plot Mean ± 95 CI
    axis = ax[row]
    sns.lineplot(data=df, x='day', y='steps', hue=hue, ax=axis)
    axis.axvline(x=0, linestyle='--', color='gray')
    axis.set_title(f"All Extremity: Mean ± 95 CI", fontdict=fontdict)
    axis.set_xlabel('Time [days]', fontdict=fontdict)
    axis.set_ylabel('Steps [%]', fontdict=fontdict)
    axis.legend(title=title)


fig.tight_layout()
fig.savefig(path_out/'steps_hue.png', dpi=300)