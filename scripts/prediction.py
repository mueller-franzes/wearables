from pathlib import Path 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import numpy as np 
from tabpfn import TabPFNClassifier

from utils import plot_roc_curve

# -------------- Settings -----------
np.random.seed(0)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.axisbelow'] = True
fontdict = {'fontweight':'bold'}
path_out = Path('results/not_normalized')
path_out.mkdir(exist_ok=True)

# ------------- Read Data ------------
df_lower_outcome = pd.read_excel('data/dataset_27_01_2025/Lower_Extremity_Patient_Outcomes.xlsx') 
df_upper_outcome = pd.read_excel('data/dataset_27_01_2025/Upper_Extremity_Patient_Outcomes.xlsx') 
df_lower_steps = pd.read_excel('data/dataset_27_01_2025/Lower_Extremity_Step_Count_Data.xlsx').set_index('Days').T 
df_upper_steps = pd.read_excel('data/dataset_27_01_2025/Upper_Extremity_Step_Count_Data.xlsx').set_index('Days').T

print()
assert len(df_lower_outcome) == len(df_lower_steps)
assert len(df_upper_outcome) == len(df_upper_steps)

df_outcome = pd.concat([df_lower_outcome, df_upper_outcome])
df_steps = pd.concat([df_lower_steps, df_upper_steps])
assert len(df_outcome) == len(df_steps)


# patient 128 appears in both - upper and lower - remove
df_outcome = df_outcome[~df_outcome['Patient_ID'].duplicated(keep='first')].reset_index(drop=True)
df_steps = df_steps[~df_steps.index.duplicated(keep='first')].reset_index(drop=True)


# remove patients with no pre-injury data
days = df_steps.columns
df_steps_pre = df_steps[days[days<0]]
no_pre_data = df_steps_pre.isna().all(axis=1)
df_steps = df_steps[~no_pre_data].reset_index(drop=True)
df_outcome = df_outcome[~no_pre_data].reset_index(drop=True)


# Normalize Steps (100%)
df_steps_pre = df_steps[df_steps.columns[df_steps.columns<0]]
lower = df_steps.min(axis=1).values[:, None]
# upper = df_steps.max(axis=1).values[:, None]
# upper = df_steps.quantile(0.99, axis=1).values[:, None]
# upper = df_steps_pre.quantile(0.99, axis=1).values[:, None]
upper = df_steps_pre.max(axis=1).values[:, None]
# df_steps = df_steps.clip(lower=lower, upper=upper, axis=1)  
# df_steps  = (df_steps-lower)/(upper-lower)*100
# df_steps  = df_steps/upper*100



# Merge Steps and Outcome  
df = pd.merge(df_steps, df_outcome, how='inner', left_index=True, right_index=True)
df['PROM_12_>Q3'] = (df['PROM_12'] > df['PROM_12'].quantile(0.75)).astype(int)
df['PROM_12_>Q2'] = (df['PROM_12'] > df['PROM_12'].quantile(0.50)).astype(int)
df['PROM_12_>Q1'] = (df['PROM_12'] > df['PROM_12'].quantile(0.25)).astype(int)
df['PROM_12_<Q1'] = (df['PROM_12'] < df['PROM_12'].quantile(0.25)).astype(int)

days_pre = days[days<0]
df_steps_pre = df[days_pre] 
df_steps_week_6 = df[range(7*6, 7*7)]
df['recovered'] = df_steps_week_6.mean(axis=1) >= df_steps_pre.mean(axis=1)*0.5


# ------------ Step 1: Select features and target ---------
# Select features and target
targets = [ 
    ('Return_to_work', "Return to work\nbased on activity", days ),
    ('Return_to_work', "Return to work\nbased on pre-injury activity", days_pre ),
    ('Return_to_work', "Return to work\nbased on pre-injury and 1 week post-injury activity", days[days<1*7]),
    ('PROM_12_>Q3', "PROM 12 > Q3\nbased on activity", days),
    ('PROM_12_>Q2', "PROM 12 > Q2\nbased on activity", days),
    ('PROM_12_>Q1', "PROM 12 > Q1\nbased on activity", days),
    ('PROM_12_<Q1', "PROM 12 < Q1\nbased on activity", days),
    ('recovered', ">50% pre-injury activity after 6 weeks\nbased on pre-injury activity", days_pre),
    ('recovered', ">50% pre-injury activity after 6 weeks\nbased on pre-injury and 1 week post-injury activity", days[days<1*7]),
]


for target, title, features in targets:
    df_clean = df.dropna(subset=target) # Drop rows with missing target 

    # Features and target variable
    X = df_clean[features]
    y = df_clean[target]


    # ----------------------- Try to estimate recovery -----------
    model = Pipeline([
        # ('scaler', StandardScaler()),
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')), 
        ('classifier', GradientBoostingClassifier(random_state=0)),
        # ('classifier', TabPFNClassifier(random_state=0))
    ])
    loo = LeaveOneOut()
    results = {'GT':[], 'NN':[], 'NN_pred':[]}
    for train_index, test_index in tqdm(loo.split(X), total=len(X)):
        X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
        y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train_cv, y_train_cv)
        prediction = model.predict(X_test_cv)
        probability = model.predict_proba(X_test_cv)
        results['GT'].append(y_test_cv.iloc[0])
        results['NN'].append(prediction[0])
        results['NN_pred'].append(probability[0,1])

    df_results = pd.DataFrame(results)
    df_results.to_csv(path_out/'results.csv')


    # --------- Evaluate the model -----------------------------
    gt = df_results['GT']
    pred = df_results['NN']
    pred_prob = df_results['NN_pred']

    # Statistics 
    print("Accuracy Score:", accuracy_score(gt, pred))
    print("Classification Report:")
    print(classification_report(gt, pred))

    # ROC-AUC 
    fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(6,6)) 
    tprs, fprs, auc_val, thrs, opt_idx, cm = plot_roc_curve(gt, pred_prob, axis, fontdict=fontdict)
    axis.set_title(f"Estimating: {title}", fontdict=fontdict)
    fig.tight_layout()
    filename = title.replace("\n", " ")
    fig.savefig(path_out/f'roc_auc_{filename}.png', dpi=300)