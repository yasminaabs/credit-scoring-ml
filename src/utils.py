"""
utils.py
========
Fonctions d'orchestration pour le TP - Partie I, Q8

La fonction pipeline_generation_train_test_split() enchaine automatiquement
les etapes vues de Q1 a Q7 et sauvegarde le pipeline final dans models/.

Auteurs : ABBAS Yasmina, GHERZOULI Maria Sara, KESRAOUI Sarah
"""

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Dossier models/ a la racine du projet (src/../models)
MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'

CRITERES = ('precision', 'recall')


# ==============================================================================
# CRITERE DE PERFORMANCE (commun au notebook et a l'orchestration)
# ==============================================================================

def score_final(y_true, y_pred, critere='precision'):
    """
    Score final du TP = (Accuracy + critere) / 2

    La classe positive (1) est le credit accorde (bon payeur), la classe 0 le defaillant :
    - 'precision' = TP / (TP + FP) : limite les credits accordes a des defaillants (risque)
    - 'recall'    = TP / (TP + FN) : limite les bons clients refuses a tort (manque a gagner)
    """
    if critere not in CRITERES:
        raise ValueError(f"critere doit valoir 'precision' ou 'recall' (recu : {critere!r})")

    acc = accuracy_score(y_true, y_pred)
    if critere == 'precision':
        crit = precision_score(y_true, y_pred, zero_division=0)
    else:
        crit = recall_score(y_true, y_pred, zero_division=0)
    return (acc + crit) / 2


def make_score_final_scorer(critere='precision'):
    """Scorer sklearn (GridSearchCV, cross_validate...) base sur score_final()"""
    return make_scorer(score_final, critere=critere)


def make_column_selector(indices):
    """
    Etape de pipeline qui ne garde que les colonnes `indices`.

    Sert a reproduire dans le pipeline exactement les variables retenues
    par l'importance RandomForest (et non un autre critere de selection).
    """
    return ColumnTransformer([('variables_retenues', 'passthrough', [int(i) for i in indices])],
                             remainder='drop')


# ==============================================================================
# FONCTION PRINCIPALE : pipeline_generation_train_test_split
# ==============================================================================

def pipeline_generation_train_test_split(X, y, test_size=0.5, random_state=1, critere='precision',
                                         models_dir=None, filename='final_pipeline_auto.pkl',
                                         verbose=True):
    """
    Fonction d'orchestration complete - fait TOUT automatiquement

    Parametres :
    -----------
    X, y : arrays
        Donnees completes
    test_size : float
        Proportion du test set
    random_state : int
        Graine aleatoire
    critere : str
        'precision' ou 'recall' : score final = (Accuracy + critere) / 2
    models_dir : str ou Path
        Dossier de sauvegarde du pipeline (par defaut : models/ a la racine du projet)
    filename : str
        Nom du fichier pickle
    verbose : bool
        Affichage detaille (texte et graphique)

    Retourne :
    ---------
    final_pipeline : Pipeline
        Pipeline sklearn entraine
    results : dict
        Dictionnaire avec tous les resultats
    """

    def titre(texte):
        if verbose:
            print("\n" + "="*70)
            print(texte.center(70))
            print("="*70)

    titre("DEMARRAGE DU WORKFLOW COMPLET")

    results = {'critere': critere}

    # ===== ETAPE 1 : SPLIT =====
    titre("ETAPE 1/7 : DIVISION TRAIN/TEST")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if verbose:
        print(f"\n Split effectue : Train={X_train.shape[0]}, Test={X_test.shape[0]}")

    results['X_train'] = X_train
    results['X_test'] = X_test
    results['y_train'] = y_train
    results['y_test'] = y_test

    # ===== ETAPE 2 : COMPARAISON MODELES =====
    titre("ETAPE 2/7 : COMPARAISON MODELES")

    classifiers_base = {
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'MLP (40,20)': MLPClassifier(hidden_layer_sizes=(40, 20),
                                      random_state=random_state, max_iter=1000)
    }

    best_scores = {}

    for name, clf in classifiers_base.items():
        clf.fit(X_train, y_train)
        score = score_final(y_test, clf.predict(X_test), critere)

        best_scores[name] = score

        if verbose:
            print(f"   {name:20s} : {score:.4f}")

    best_model_name = max(best_scores, key=best_scores.get)
    best_model_score = best_scores[best_model_name]
    best_clf = classifiers_base[best_model_name]

    if verbose:
        print(f"\n Meilleur : {best_model_name} ({best_model_score:.4f})")

    results['best_model_base'] = best_model_name
    results['scores_base'] = best_scores

    # Graphique
    if verbose:
        models = list(best_scores.keys())
        scores = list(best_scores.values())

        plt.figure(figsize=(8, 5))
        bars = plt.bar(models, scores, color=['green' if m == best_model_name else 'steelblue' for m in models],
                       alpha=0.7, edgecolor='black')
        plt.title("Comparaison des modeles de base", fontweight='bold')
        plt.ylabel(f"Score (Accuracy + {critere.capitalize()}) / 2", fontweight='bold')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)

        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{score:.4f}', ha='center', fontweight='bold')

        plt.tight_layout()
        plt.show()

    # ===== ETAPE 3 : TEST NORMALISATION =====
    titre("ETAPE 3/7 : TEST NORMALISATION")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf_norm = clone(best_clf)
    clf_norm.fit(X_train_scaled, y_train)
    score_norm = score_final(y_test, clf_norm.predict(X_test_scaled), critere)

    use_normalization = score_norm > best_model_score

    if verbose:
        print(f"\n   Sans normalisation : {best_model_score:.4f}")
        print(f"   Avec normalisation : {score_norm:.4f}")
        print(f"\n   Decision : {'AVEC normalisation' if use_normalization else 'SANS normalisation'}")

    results['use_normalization'] = use_normalization
    results['score_normalized'] = score_norm

    X_train_prep = X_train_scaled if use_normalization else X_train
    X_test_prep = X_test_scaled if use_normalization else X_test

    # ===== ETAPE 4 : SELECTION FEATURES =====
    titre("ETAPE 4/7 : SELECTION FEATURES")

    # Classement des variables par importance RandomForest
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf.fit(X_train_prep, y_train)

    sorted_idx = np.argsort(rf.feature_importances_)[::-1]

    if verbose:
        print(f"\n   Top 3 features : {sorted_idx[:3].tolist()}")

    # Nombre optimal de variables (les n plus importantes) selon l'accuracy
    best_n_features = X_train.shape[1]
    best_acc_features = 0

    for n in range(1, X_train.shape[1] + 1):
        selected = sorted_idx[:n]

        clf_temp = clone(best_clf)
        clf_temp.fit(X_train_prep[:, selected], y_train)
        acc = clf_temp.score(X_test_prep[:, selected], y_test)

        if acc > best_acc_features:
            best_acc_features = acc
            best_n_features = n

    selected_features = sorted_idx[:best_n_features]

    if verbose:
        print(f"   Optimal : {best_n_features} features (acc={best_acc_features:.4f})")

    results['n_features_optimal'] = best_n_features
    results['sorted_features'] = sorted_idx
    results['selected_features'] = selected_features

    # ===== ETAPE 5 : OPTIMISATION =====
    titre("ETAPE 5/7 : OPTIMISATION HYPERPARAMETRES")

    if 'MLP' in best_model_name:
        base_estimator = MLPClassifier(random_state=random_state, max_iter=1000)
        param_grid = {
            'hidden_layer_sizes': [(40, 20), (60, 30)],
            'alpha': [0.001, 0.01],
            'learning_rate_init': [0.01]
        }
    elif 'KNN' in best_model_name:
        base_estimator = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }
    else:
        base_estimator = DecisionTreeClassifier(random_state=random_state)
        param_grid = {
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        }

    grid = GridSearchCV(base_estimator, param_grid, scoring=make_score_final_scorer(critere),
                        cv=3, n_jobs=-1, verbose=0)

    if verbose:
        print(f"\n   GridSearch en cours...")

    # Optimisation sur les memes variables que celles du pipeline final
    grid.fit(X_train_prep[:, selected_features], y_train)

    if verbose:
        print(f"   Meilleurs params : {grid.best_params_}")
        print(f"   Score CV : {grid.best_score_:.4f}")

    results['best_params'] = grid.best_params_
    results['best_estimator'] = grid.best_estimator_

    # ===== ETAPE 6 : PIPELINE FINAL =====
    titre("ETAPE 6/7 : CREATION PIPELINE")

    steps = []

    if use_normalization:
        steps.append(('scaler', StandardScaler()))
        if verbose:
            print("   - StandardScaler ajoute")

    # Memes variables qu'a l'etape 4 (importance RandomForest)
    steps.append(('feature_selection', make_column_selector(selected_features)))
    if verbose:
        print(f"   - Selection des {best_n_features} variables les plus importantes ajoutee")

    steps.append(('classifier', clone(grid.best_estimator_)))
    if verbose:
        print(f"   - {type(grid.best_estimator_).__name__} ajoute")

    final_pipeline = Pipeline(steps)
    final_pipeline.fit(X_train, y_train)

    if verbose:
        print(f"\n   Pipeline cree avec {len(steps)} etapes")

    # ===== ETAPE 7 : SAUVEGARDE =====
    titre("ETAPE 7/7 : SAUVEGARDE ET EVALUATION")

    y_pred_final = final_pipeline.predict(X_test)

    acc_final = accuracy_score(y_test, y_pred_final)
    rec_final = recall_score(y_test, y_pred_final, zero_division=0)
    prec_final = precision_score(y_test, y_pred_final, zero_division=0)
    score_test = score_final(y_test, y_pred_final, critere)

    if verbose:
        print(f"\n SCORES FINAUX :")
        print(f"   Accuracy  : {acc_final:.4f} ({acc_final*100:.2f}%)")
        print(f"   Recall    : {rec_final:.4f} ({rec_final*100:.2f}%)")
        print(f"   Precision : {prec_final:.4f} ({prec_final*100:.2f}%)")
        print(f"   Score Final : {score_test:.4f} ({score_test*100:.2f}%)")

    results['final_scores'] = {
        'accuracy': acc_final,
        'recall': rec_final,
        'precision': prec_final,
        'score_final': score_test
    }

    models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = models_dir / filename

    with open(pipeline_path, 'wb') as f:
        pickle.dump(final_pipeline, f)

    if verbose:
        file_size = pipeline_path.stat().st_size / 1024
        print(f"\n Pipeline sauvegarde : {pipeline_path.parent.name}/{pipeline_path.name} ({file_size:.1f} KB)")
        print("\n" + "="*70)
        print("WORKFLOW TERMINE AVEC SUCCES !".center(70))
        print("="*70)

    results['pipeline_path'] = str(pipeline_path)
    results['pipeline'] = final_pipeline

    return final_pipeline, results
