import numpy as np
import pandas as pd
import random
import os

# -- Data fetch: pip install ucimlrepo if needed
from ucimlrepo import fetch_ucirepo

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# XGBoost
from xgboost import XGBClassifier

# SHAP & plotting
import matplotlib.pyplot as plt

# TensorFlow/Keras for NN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Copula utilities
from scipy.stats import kendalltau, spearmanr, rankdata
from scipy.optimize import bisect

# 1. Reproducibility
def set_seed(seed=123):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
set_seed(123)

# 2. Utility functions
def to_pseudo_observations(arr):
    return rankdata(arr) / (len(arr) + 1)

def compute_summary_features(U, V):
    tau, _ = kendalltau(U, V)
    rho, _ = spearmanr(U, V)
    tail = np.mean((U > 0.95) & (V > 0.95))
    corr = np.corrcoef(U, V)[0,1]
    return np.array([tau, rho, tail, corr])

# 3. Copula & NN-based A2 θ estimation
def phi_A2(t, θ): return ((1.0/t)*(1.0 - t)**2)**θ
def dphi_A2(t, θ):
    g = (1.0 - t)**2 / t
    num = -(1.0 - t)*(1.0 + t)
    den = t*t
    return θ*(g**(θ-1))*(num/den)
def phi_A2_inv(y, θ):
    a = 2.0 + y**(1.0/θ)
    return (a - np.sqrt(a*a - 4.0)) / 2.0

def K_fun(x, φ, dφ, θ):
    return x - φ(x, θ)/(dφ(x, θ)+1e-15)
def K_inv(t, φ, dφ, θ):
    return bisect(lambda x: K_fun(x, φ, dφ, θ)-t, 1e-14, 1-1e-14, xtol=1e-9)

def sample_arch(φ, dφ, φ_inv, θ, n=2000):
    s, t = np.random.rand(n), np.random.rand(n)
    U, V = np.empty(n), np.empty(n)
    for i in range(n):
        w = K_inv(t[i], φ, dφ, θ)
        φw = φ(w, θ)
        U[i] = φ_inv(s[i]*φw, θ)
        V[i] = φ_inv((1-s[i])*φw, θ)
    return U, V

# 3. Copula & NN-based A2 θ estimation 

# First modify the training data generation to match theta range
def gen_train_data(n_samp=300, samp_size=2000):  # Increased n_samp from 200 to 300
    funcs = (phi_A2, dphi_A2, phi_A2_inv)
    Xs, ys = [], []
    thetas = np.linspace(1.0, 20.0, n_samp)  # Changed upper bound from 10.0 to 20.0
    for θ in thetas:
        U, V = sample_arch(*funcs, θ, samp_size)
        Xs.append(compute_summary_features(U,V))
        ys.append(θ)
    return np.vstack(Xs), np.array(ys)


# Generate training data - produces 4 summary features
X_sim, y_sim = gen_train_data(n_samp=300, samp_size=2000)

# KEY CHANGE: Fit scaler ONLY to the 4 raw features (not to one-hot encoded version)
scaler_sim = StandardScaler().fit(X_sim)  # Fit to 4D features only

# Prepare expanded training data (scale first, then add one-hot)
X_scaled = scaler_sim.transform(X_sim)  # Scale the 4 features
a2_one_hot = np.array([0, 0, 0, 0, 1])
Xs_expanded = np.hstack([X_scaled, np.tile(a2_one_hot, (X_sim.shape[0], 1))])  # 9D

# Build NN 
nn = keras.Sequential([
    layers.Input(shape=(9,)),  # Explicit 9D input
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='softplus'),
    layers.Lambda(lambda x: x + 1.0)  # Ensures theta >= 1
])

# Compile and train
nn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='mse')
nn.fit(Xs_expanded, y_sim, batch_size=32, epochs=30, verbose=0)


def estimate_θ_A2(pair):
    U = to_pseudo_observations(pair[:, 0])
    V = to_pseudo_observations(pair[:, 1])
    feats = compute_summary_features(U, V)  # 4 features

    # Process EXACTLY like training data:
    scaled_feats = scaler_sim.transform(feats.reshape(1, -1))  # Scale 4D features
    input_vector = np.hstack([scaled_feats, [[0, 0, 0, 0, 1]]])  # Add raw one-hot

    θ_hat = nn.predict(input_vector, verbose=0)[0, 0]
    return float(θ_hat)
# -------------------------------
# 4. Load & Preprocess CDC Diabetes Data
# -------------------------------
from ucimlrepo import fetch_ucirepo

cdc = fetch_ucirepo(id=891)
X   = cdc.data.features
y_df = cdc.data.targets            # this is a DataFrame
y    = y_df.to_numpy().ravel()     # now a 1D numpy array
y    = (y > 0).astype(int)         # binary 0/1

# Drop any columns that are all missing or constant
X = X.dropna(axis=1, how='all')
X = X.loc[:, X.nunique() > 1]

# 5. Feature selection
# A2-based
Xp = X.apply(lambda c: to_pseudo_observations(c.values))
yp = to_pseudo_observations(y)
scores = []
for f in Xp.columns:
    theta = estimate_θ_A2(np.column_stack((Xp[f],yp)))
    lam = 2 - 2**(1/(2*theta))
    scores.append((f,lam))
df_a2 = pd.DataFrame(scores,columns=['Feature','Lambda_U']).dropna()
top_a2 = df_a2.nlargest(5,'Lambda_U')['Feature'].tolist()

# MI-based
mi = mutual_info_classif(X,y,random_state=123)
df_mi = pd.Series(mi,index=X.columns).sort_values(ascending=False)
top_mi = df_mi.head(5).index.tolist()

# GA Wrapper (pop=10, gens=5)
def evaluate_subset(indices):
    if not indices:
        return 0
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=123)
    return cross_val_score(rf, X.iloc[:, indices], y, cv=5).mean()

def ga(p, k, pop_size=10, gens=5):
    population = [random.sample(range(p), k) for _ in range(pop_size)]
    for _ in range(gens):
        fitness = [evaluate_subset(ind) for ind in population]
        new_population = []
        while len(new_population) < pop_size:
            # Tournament selection
            i, j = random.sample(range(pop_size), 2)
            p1 = population[i] if fitness[i] > fitness[j] else population[j]
            u, v = random.sample(range(pop_size), 2)
            p2 = population[u] if fitness[u] > fitness[v] else population[v]
            # Crossover
            half = k // 2
            child = list(set(p1[:half] + p2[half:]))
            # Repair
            while len(child) < k:
                idx = random.randrange(p)
                if idx not in child:
                    child.append(idx)
            new_population.append(child)
        population = new_population

    fitness = [evaluate_subset(ind) for ind in population]
    best_idx = int(np.argmax(fitness))
    return population[best_idx]

# Now call it
top_ga_idx = ga(X.shape[1], 5, pop_size=10, gens=5)
top_ga = X.columns[top_ga_idx].tolist()

print("\nTop 5 features by A2 upper-tail dependency:")
print(top_a2)

print("\nTop 5 features by Mutual Information:")
print(top_mi)

print("\nTop 5 features by Genetic Algorithm:")
print(top_ga)

# 6. Classification & metrics
classifiers = {
 'RF':RandomForestClassifier(n_jobs=-1,random_state=123),
 'XGB':XGBClassifier(use_label_encoder=False,eval_metric='logloss',random_state=123),
 'LR':LogisticRegression(max_iter=1000,random_state=123),
 'GB':GradientBoostingClassifier(random_state=123)
}

sets = {'All':X,'A2':X[top_a2],'MI':X[top_mi],'GA':X[top_ga]}
results=[]
for name,df in sets.items():
    Xt,Xt2,yt,yt2=train_test_split(df,y,test_size=0.2,random_state=123,stratify=y)
    scaler_lr=StandardScaler().fit(Xt)
    Xl, Xl2 = scaler_lr.transform(Xt), scaler_lr.transform(Xt2)
    for m,clf in classifiers.items():
        if m=='LR':
            clf.fit(Xl,yt)
            y_pred=clf.predict(Xl2)
            y_pr = clf.predict_proba(Xl2)[:,1]
        else:
            clf.fit(Xt,yt)
            y_pred=clf.predict(Xt2)
            y_pr = clf.predict_proba(Xt2)[:,1]
        cr=classification_report(yt2,y_pred,output_dict=True)
        auc=roc_auc_score(yt2,y_pr)
        results.append({
            'Set':name,'Model':m,
            'Accuracy':cr['accuracy'],'Precision':cr['weighted avg']['precision'],
            'Recall':cr['weighted avg']['recall'],'F1':cr['weighted avg']['f1-score'],'AUC':auc
        })
df_res=pd.DataFrame(results)
df_res.to_csv("feature_selection_comparison_results.csv",index=False)
print(df_res)

# -------------------------------
# 7. Interpretability: Permutation Importance
# -------------------------------
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Train RF on A2 features
Xtr, Xte, ytr, yte = train_test_split(X[top_a2], y, test_size=0.2, random_state=123, stratify=y)
rf = RandomForestClassifier(n_jobs=-1, random_state=123).fit(Xtr, ytr)

# Permutation importance
perm = permutation_importance(
    rf, Xte, yte,
    n_repeats=20,
    random_state=123,
    n_jobs=-1
)
importances = pd.Series(perm.importances_mean, index=Xte.columns).sort_values(ascending=False)

# Plot
plt.figure(figsize=(6,4))
importances.plot.barh()
plt.gca().invert_yaxis()
plt.xlabel("Mean decrease in accuracy")
plt.title("Permutation Importances (RF on A2 features)")
plt.tight_layout()
plt.show()















