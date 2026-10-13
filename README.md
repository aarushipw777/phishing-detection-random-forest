# URL-Based Phishing Detection Using Machine Learning

A machine learning mini project that classifies a website as **phishing** or
**legitimate** from 21 feature values, using a **Random Forest** model and a
**Flask** web interface.

---

## 1. The Problem

**Phishing** is a cyber-attack where criminals build fake websites that imitate
trusted ones (a bank, an email login, a shopping site) to steal usernames,
passwords and card details. Victims often cannot tell the fake from the real site.

Traditional defences use fixed blacklists and hand-written rules. These struggle
because attackers constantly create new sites and change their tricks, so a list
is out of date almost as soon as it is written.

### Problem statement

> Given 21 feature values that describe a website (for example whether it uses
> HTTPS, whether it uses an IP address instead of a domain name, or whether it has
> unusual link patterns), automatically classify the website as **phishing** or
> **legitimate**.

In machine learning terms this is a **binary classification** problem under
**supervised learning**: the model learns from examples that already have
the correct label.

### Objectives

- Prepare a public phishing dataset and train a Random Forest classifier on it
- Evaluate the model on unseen data with accuracy, precision, recall, F1-score and a confusion matrix
- Save the trained model and serve its predictions through a Flask web app

---

## 2. The Approach (How We Reached a Solution)

Instead of writing rules by hand, we let a model **learn the warning signs** from
thousands of websites already labelled as phishing or legitimate.

```
Dataset -> Clean -> Split (80/20) -> Select features -> Train Random Forest
        -> Evaluate on unseen data -> Save model -> Flask web app
```

| Step | What we did | Why |
|---|---|---|
| 1. Inspect | Checked shape, data types, missing values, duplicates, class balance | Know the data before using it |
| 2. Clean | Dropped the `Index` column and removed 5,205 duplicate rows | `Index` is only a row number; duplicates could leak into both train and test sets and inflate the score |
| 3. Define X and y | X = the 30 features, y = the class label | The model needs inputs and the answer it must predict |
| 4. Split | 80% training (4,679 rows) / 20% test (1,170 rows), stratified, `random_state=42` | The test set is kept unseen to measure real performance; stratified keeps the same phishing/legitimate ratio in both parts |
| 5. Feature selection | Ranked features by absolute correlation with the label and kept the strongest 21 (using training data only) | Removes weak features; using only training data avoids test-set leakage |
| 6. Train | Random Forest with 100 trees (`n_estimators=100`, `random_state=42`) | Strong default performance on this kind of data |
| 7. Evaluate | Scored on the 1,170 unseen test rows, phishing as the positive class | Honest estimate of how it behaves on new sites |
| 8. Save and serve | Saved the model with `pickle`; Flask loads it and predicts from form input | Train once, use many times |

Feature scaling, outlier removal and oversampling were **not** used: the features
are small coded values, tree-based models do not need scaling, and the classes
were nearly balanced.

---

## 3. Dataset

Kaggle **"Phishing Website Detector"** (`phishing.csv`) by eswarchandt:
https://www.kaggle.com/eswarchandt/phishing-website-detector

- Raw file: 11,054 rows x 32 columns (`Index`, 30 features, 1 class label)
- Every feature is an integer coded **-1, 0 or 1**; there are no missing values
- After removing duplicates: **5,849 rows**, of which 3,019 are phishing (label `-1`, 51.6%) and 2,830 are legitimate (label `1`, 48.4%). The classes are nearly balanced.
- We checked the label direction against the data: legitimate rows have higher average values on features such as `HTTPS` and `AnchorURL`

Each feature describes one property of a website (for example `HTTPS`, `AnchorURL`,
`SubDomains`, `PrefixSuffix-`, `WebsiteTraffic`). The coding generally follows the
pattern that `1` looks safe, `0` is suspicious or unclear, and `-1` looks like phishing.

The dataset is not included in this repo. To rerun the notebook, download it and
place it in a `data/` folder.

**Features dropped** during selection: `Redirecting//`, `Favicon`, `NonStdPort`,
`InfoEmail`, `WebsiteForwarding`, `StatusBarCust`, `DisableRightClick`,
`UsingPopupWindow`, `IframeRedirection`.

---

## 4. The Algorithm: Random Forest

A **decision tree** asks a series of yes/no questions about the features (for
example "is HTTPS = -1?") and ends at a verdict. A single tree can memorise its
training data, which is called **overfitting**.

A **Random Forest** builds many trees (100 here). Each tree is trained on a
random sample of the rows and considers a random subset of features at each
split. The final answer is the **majority vote** of all trees. Because the trees
differ, their individual mistakes tend to cancel out.

Why it suits this project:
- handles small integer-coded features without scaling
- performs well with default settings
- reports which features mattered most (feature importance)

---

## 5. Technologies Used

| Tool | Role in this project |
|---|---|
| **Python** | Programming language for everything |
| **pandas** | Loading the CSV, cleaning, removing duplicates, building tables |
| **NumPy** | Numerical operations on arrays |
| **scikit-learn** | Train/test split, Random Forest, and evaluation metrics |
| **matplotlib / seaborn** | Charts such as the confusion matrix and feature importance |
| **Jupyter Notebook** | Interactive environment for the training and evaluation code (`train_model.ipynb`) |
| **pickle** | Saves the trained model to `phishing_rf.pkl` so it can be reloaded without retraining |
| **Flask** | Lightweight web framework that serves the form and shows predictions |
| **HTML template** | `templates/index.html`, the page users see |
| **venv** | Virtual environment keeping this project's libraries separate from other Python projects |
| **JSON files** | `feature_options.json` (form choices) and `examples.json` (sample rows) |

---

## 6. System Architecture

The machine learning part runs **once** and produces one file: the saved model.
The web part only **loads** that file.

```
[Training - runs once]                 [Web app - runs on demand]
phishing.csv -> train_model.ipynb      Browser form (21 values)
              -> phishing_rf.pkl   ->  app.py loads model
                                       -> one-row table, same column order as training
                                       -> model.predict()
                                       -> "Phishing" or "Legitimate" shown on page
```

Flask itself contains no intelligence. The learned behaviour lives entirely in the
trained Random Forest.

---

## 7. Results

Evaluated on **1,170 unseen test websites** (604 phishing, 566 legitimate).
Phishing is the positive class.

| Metric | Value |
|---|---|
| Accuracy | 0.9385 |
| Precision (phishing) | 0.9433 |
| Recall (phishing) | 0.9371 |
| F1-score (phishing) | 0.9402 |
| Training accuracy (for comparison) | 0.9885 |

### Confusion matrix

| | Predicted: Phishing | Predicted: Legitimate |
|---|---|---|
| **Actual: Phishing (604)** | 566 (True Positive) | 38 (False Negative) |
| **Actual: Legitimate (566)** | 34 (False Positive) | 532 (True Negative) |

### What the metrics mean

- **Accuracy**: share of all predictions that were correct (1,098 of 1,170)
- **Precision**: of the sites flagged as phishing, how many really were
- **Recall**: of all real phishing sites, how many were caught (566 of 604)
- **F1-score**: a single number balancing precision and recall
- **False negative (38)**: a phishing site called safe. This is the **dangerous** error.
- **False positive (34)**: a safe site flagged as phishing. This is the **nuisance** error.

The two error counts are almost equal, so the model is not strongly biased
towards either mistake. Training accuracy is about five points above test
accuracy; this mild overfitting is normal for a Random Forest, and the **test**
figure is the one that estimates real performance.

### Most important features

`HTTPS` (0.333) and `AnchorURL` (0.234) together account for about 57% of the
total importance, followed by `SubDomains` (0.063), `PrefixSuffix-` (0.061) and
`WebsiteTraffic` (0.055). Feature importance shows what the model relied on, not
what *causes* phishing.

The Flask app was tested with example rows taken from the test set. No real
phishing website was visited.

---

## 8. Files

| File | Purpose |
|---|---|
| `train_model.ipynb` | Data preparation, training, evaluation |
| `app.py` | Flask web application |
| `templates/index.html` | Web page for the app |
| `phishing_rf.pkl` | Trained Random Forest model |
| `feature_options.json`, `examples.json` | Helper files for the web form |
| `requirements.txt` | Python libraries and versions used |

---

## 9. How to Run (Windows)

```
py -m venv phishing-venv
phishing-venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000 in your browser. Keep the terminal open while
using the app and press `Ctrl + C` to stop it.

To explore the notebook, run `jupyter notebook` from the same activated
environment and open `train_model.ipynb`.

---

## 10. Limitations

- The app takes **feature values from the dataset**, not a raw URL. Many features (such as PageRank, GoogleIndex and AgeofDomain) need page content or outside lookups.
- Results come from a **single train/test split** of 1,170 rows, so they carry an uncertainty of roughly one to two percentage points.
- The dataset reflects the phishing techniques of its time. For example, many phishing sites now use HTTPS, so HTTPS is a weaker warning sign than it once was.
- The model **does not keep learning** after training; it must be retrained on new data.
- Only load `.pkl` files from sources you trust.

## 11. Future Scope

- Extract features automatically from a pasted URL so users only need to provide a link
- Compare other models (such as Gradient Boosting) using cross-validation, and train on newer datasets
- Integrate the model into an email filter or browser extension

## References

1. Kaggle, "Phishing Website Detector" dataset (eswarchandt)
2. Reference project studied: B. Sumanjali et al., "URL-Based Phishing Detection using Machine Learning" (GitHub). The preprocessing, model, results and application here were produced independently.
3. scikit-learn and Flask documentation

## Author

Aarushi Tiwari
