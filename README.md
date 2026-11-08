# Devoir Machine Learning : Support Vector Machines (SVC & SVR) sur Auto MPG
**Master Sciences et Techniques : Ingénierie des Données et IA (MST IDIA)**  
**Faculté des Sciences et Techniques de Tanger (FSTT)**  
**Module :** Apprentissage Automatique — **Enseignant :** M. AIT KBIR  
**Auteur :** Abajja Mohammed

---

## 1. Introduction et Objectifs

Ce projet a pour objectif la modélisation et l'approximation de la consommation de carburant de véhicules automobiles à partir du jeu de données classique **Auto MPG** (disponible via Seaborn).

Le problème est abordé sous deux angles complémentaires en utilisant les machines à vecteurs de support (SVM) :
1. **Une tâche de classification multiclasse (SVC)** : discrétisation de la consommation continue en trois classes sémantiques (*Élevée*, *Moyenne*, *Faible*), avec implémentation *From Scratch* via programmation quadratique duale (noyaux linéaire et RBF) et comparaison avec `sklearn.svm.SVC`.
2. **Une tâche de régression (SVR)** : prédiction directe de la valeur continue `mpg` au moyen d'un modèle linéaire intégrant la fonction de perte $\epsilon$-insensible (*$\epsilon$-insensitive loss*), étude de l'impact de $\epsilon$ sur la parcimonie et la robustesse au bruit, et comparaison avec `sklearn.svm.SVR`.

---

## 2. Pipeline de Prétraitement des Données

Le dataset original contient des variables continues, discrètes et catégorielles. Les choix de prétraitement suivants ont été opérés et justifiés :

1. **Traitement des valeurs manquantes :**
   * La variable `horsepower` comportait 6 valeurs manquantes (`NaN`). Conformément à la consigne indiquant un échantillon final de 392 observations ($398 - 6 = 392$), ces 6 lignes ont été supprimées (`dropna()`).
2. **Élimination de la variable non informative :**
   * La colonne textuelle `name` contient des dénominations de modèles uniques (forte cardinalité sans structure ordinale). Elle a été exclue car elle n'apporte aucun pouvoir de généralisation statistique pour un classifieur géométrique.
3. **Encodage de la variable catégorielle nominale :**
   * La variable `origin` (1: USA, 2: Europe, 3: Japan) est strictement nominale (aucun ordre hiérarchique naturel). Un encodage ordinal induirait une fausse distance géométrique euclidienne. Nous avons donc appliqué un **One-Hot Encoding** (`pd.get_dummies` avec `drop_first=True`) aboutissant à un espace de description de **9 variables explicatives** (`displacement`, `horsepower`, `weight`, `acceleration`, `model_year`, `cylinders`, `origin_europe`, `origin_japan`).
4. **Standardisation (`StandardScaler`) :**
   * Les SVM reposent sur la maximisation de la marge géométrique et le calcul de produits scalaires / distances euclidiennes ($\|X_i - X_j\|^2$). Sans mise à l'échelle, les variables à forte variance (ex. `weight` $\approx 3000$) écraseraient artificiellement celles à faible amplitude (ex. `acceleration` $\approx 15$).
   * Le centrage-réduction est ajusté exclusivement sur le jeu d'entraînement (`X_train`) et propagé sur le jeu de test (`X_test`) afin de prévenir toute fuite d'information (*data leakage*).

---

## 3. Tâche de Classification : SVC

### 3.1 Stratégie de Discrétisation de la Cible `mpg`

En analyse automobile américaine, la métrique **MPG (Miles Per Gallon)** mesure l'efficacité énergétique : un MPG élevé signifie que le véhicule parcourt une distance supérieure par unité d'essence, ce qui correspond à une **consommation faible**.
* **Classe 0 :** Consommation Élevée (MPG faible)
* **Classe 1 :** Consommation Moyenne
* **Classe 2 :** Consommation Faible (MPG élevé)

Deux stratégies de découpage ont été évaluées empiriquement :

| Stratégie | Méthode Pandas | Intervalles obtenus (MPG) | Distribution des classes |
| :--- | :--- | :--- | :--- |
| **Largeurs égales** | `pd.cut` | $[9.0, 21.5[, [21.5, 34.1[, [34.1, 46.6]$ | Élevée : 182, Moyenne : 168, **Faible : 42** |
| **Quantiles / Terciles** | `pd.qcut` | $[9.0, 18.7[, [18.7, 26.9[, [26.9, 46.6]$ | **Élevée : 131, Moyenne : 130, Faible : 131** |

![Comparaison des discrétisations](imgs/discretisation.png)

> **Justification du choix final :**  
> La distribution empirique de `mpg` est asymétrique à droite. L'approche par largeurs égales (`pd.cut`) engendre un fort déséquilibre de classes, avec seulement 42 exemples pour la classe *Consommation Faible*. Un tel déséquilibre tire l'hyperplan séparateur au détriment de la classe minoritaire.  
> **Nous avons donc retenu la discrétisation par quantiles (`pd.qcut`)**, qui garantit une répartition équilibrée ($\approx 33.3\%$ par classe), assurant un apprentissage non biaisé et une convergence stable du solveur.

---

### 3.2 Implémentation From Scratch du SVC

Conformément au formalisme théorique du cours (Slides 9, 12 et 14), le classifieur binaire résout le problème dual de marge douce (*soft-margin*) via programmation quadratique (`cvxopt.solvers.qp`) :

$$\max_{\alpha} \left( -\frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j Y_i Y_j K(X_i, X_j) + \sum_{i=1}^N \alpha_i \right)$$

Sous les contraintes :
$$\sum_{i=1}^N \alpha_i Y_i = 0 \quad \text{et} \quad 0 \le \alpha_i \le C, \quad \forall i \in \{1, \dots, N\}$$

* **Noyaux implémentés :**
  * Linéaire : $K(X_i, X_j) = X_i^T X_j$
  * RBF (Gaussien) : $K(X_i, X_j) = \exp(-\gamma \|X_i - X_j\|^2)$ avec $\gamma = \frac{1}{d}$ (où $d$ est le nombre de descripteurs).
* **Calcul du biais optimal $w_0^*$ (Slide 10) :**
  Calculé sur les vecteurs de support ($0 < \alpha_i < C$) :
  $$w_0^* = \frac{1}{N_s} \sum_{i \in SV} \left( Y_i - \sum_{j \in SV} \alpha_j^* Y_j K(X_j, X_i) \right)$$
* **Extension multiclasse One-vs-Rest (OvR) :**  
  Trois classifieurs binaires sont entraînés ($c$ vs reste). La prédiction finale attribue l'observation à la classe maximisant la distance algébrique : $\hat{y} = \arg\max_{c} f_c(X)$.

---

### 3.3 Résultats et Comparaison avec Scikit-Learn

Sur le jeu de test ($20\%$ stratifié, $N_{test} = 79$) :

![Résultats SVC](imgs/svc_comparison.png)

| Modèle | Noyau | Implémentation | Accuracy (Test) |
| :--- | :--- | :--- | :---: |
| SVC | Linéaire | **From Scratch (CVXOPT)** | **0.784810** |
| SVC | Linéaire | Scikit-Learn (`libsvm`) | **0.784810** |
| SVC | RBF | **From Scratch (CVXOPT)** | **0.772152** |
| SVC | RBF | Scikit-Learn (`libsvm`) | **0.772152** |

> **Analyse des résultats :**  
> Les métriques d'accuracy sont **strictement identiques au 6ème chiffre après la virgule** entre notre code *From Scratch* et l'implémentation de référence de Scikit-Learn, validant mathématiquement notre formulation matricielle sous `cvxopt` et la gestion du multiclasse OvR.

---

## 4. Tâche de Régression : SVR Linéaire

### 4.1 Principe de la Perte $\epsilon$-insensible (Slide 15)

Contrairement à la régression des moindres carrés qui pénalise tout écart quadratique, le modèle SVR introduit un tube d'insensibilité de largeur $2\epsilon$ centré sur l'hyperplan de prédiction $f(X) = W^T X + w_0$.

$$\min_{W, w_0, \xi, \xi^*} \frac{1}{2} \|W\|^2 + C \sum_{i=1}^N (\xi_i + \xi_i^*)$$
$$\text{s.c.} \quad \begin{cases} Y_i - (W^T X_i + w_0) \le \epsilon + \xi_i \\ (W^T X_i + w_0) - Y_i \le \epsilon + \xi_i^* \\ \xi_i, \xi_i^* \ge 0 \end{cases}$$

La fonction de coût associée est la perte $\epsilon$-insensible :
$$L_\epsilon(y, \hat{y}) = \max(0, |y - \hat{y}| - \epsilon)$$

Les observations situées à l'intérieur du tube ont une erreur nulle ; seuls les points situés sur ou hors de la frontière constituent les **vecteurs de support**.

---

### 4.2 Résultats Comparatifs (à $\epsilon = 1.5, C = 2.0$)

![Résultats SVR](imgs/svr_comparison.png)

| Modèle | Métrique RMSE | Coefficient $R^2$ |
| :--- | :---: | :---: |
| **SVR Linéaire (From Scratch - Subgradient)** | **3.4655** | **0.7647** |
| **SVR Linéaire (Scikit-Learn - Dual QP)** | **3.2991** | **0.7867** |

> **Interprétation :**  
> Notre régresseur *From Scratch* (optimisé par descente de sous-gradient du problème primal) atteint une performance très proche du solveur dual de Scikit-Learn ($R^2 \approx 76.5\%$ vs $78.7\%$), confirmant la bonne convergence de l'algorithme.

---

### 4.3 Étude de l'Influence du Paramètre $\epsilon$

Une étude paramétrique a été menée en faisant varier $\epsilon \in [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]$ à régularisation constante ($C = 2.0$) :

![Étude paramètre Epsilon](imgs/svr_epsilon_study.png)

#### Interprétations clés :
1. **Effet sur le nombre de vecteurs de support (Parcimonie) :**  
   * À $\epsilon = 0.1$, le tube est extrêmement étroit : quasiment tous les points d'entraînement sont en dehors et deviennent des vecteurs de support (**$\approx 305$ vecteurs de support sur 313 points de train**).
   * À mesure que $\epsilon$ augmente, le tube s'élargit et englobe davantage d'échantillons. Le nombre de vecteurs de support chute drastiquement pour atteindre **moins de 30 à $\epsilon = 6.0$**. Le modèle devient hautement parcimonieux (*sparse*), réduisant considérablement la complexité mémoire et le temps d'inférence.
2. **Effet sur l'erreur et la tolérance au bruit (Compromis Biais-Variance) :**  
   * **Pour $\epsilon \in [0.1, 1.0]$ :** Le modèle pénalise les moindres écarts résiduels, ajustant sa pente sur les bruits de mesure locaux ($RMSE \approx 3.33$).
   * **Pour $\epsilon = 3.0$ :** Le modèle atteint son minimum d'erreur de test (**$RMSE \approx 3.20$**). Le tube a la largeur optimale pour absorber le bruit aléatoire sans pénaliser la tendance générale.
   * **Pour $\epsilon > 4.0$ :** Le tube devient trop lâche. Le modèle commence à sous-apprendre (*underfitting*) et ignore de réelles variations de consommation, ce qui fait bondir le RMSE à **$> 3.55$**.

---

## 5. Guide d'Exécution

### 5.1 Installation de l'environnement
```bash
pip install -r requirements.txt