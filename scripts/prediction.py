from pathlib import Path 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from utils import plot_roc_curve

# -------------- Settings -----------
plt.rcParams['font.size'] = 11
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.axisbelow'] = True
fontdict = {'fontweight':'bold'}
path_out = Path('results/')
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
df_steps = df_steps[~df_steps.index.duplicated(keep='first')]

# Normalize Steps (100%)
max_steps = df_steps.max(axis=1)
df_steps  = df_steps.div(max_steps, axis=0)*100

# Prepare data 
df_steps.columns = df_steps.columns.astype(str)
days = df_steps.columns
df_steps['Patient_ID'] = df_steps.index

# Merge Steps and Outcome  
df = pd.merge(df_steps, df_outcome, how='inner', on='Patient_ID')
df['PROM_12_Q1'] = (df['PROM_12'] <= df['PROM_12'].quantile(0.25)).astype(int)

# ------------ Step 1: Select features and target ---------
# Select features and target
features = [*days, ]

targets = [ ('Return_to_work', "Return to work"),
           ('PROM_12_Q1', "Withing Q1 of PROM 12"),
]


for target, title in targets:
    

    # Drop rows with any missing values in these columns
    df_clean = df.dropna(subset=target)
    df_clean = df_clean.fillna(-1)

    # Features and target variable
    X = df_clean[features]
    y = df_clean[target]


    # ----------------------- Try to estimate recovery -----------
    model = Pipeline([
        ('scaler', StandardScaler()), 
        ('classifier', GradientBoostingClassifier())
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
    fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(5,5)) 
    tprs, fprs, auc_val, thrs, opt_idx, cm = plot_roc_curve(gt, pred_prob, axis, fontdict=fontdict)
    axis.set_title(f"Estimating: {title}", fontdict=fontdict)
    fig.tight_layout()
    fig.savefig(path_out/f'roc_auc_{target}.png', dpi=300)