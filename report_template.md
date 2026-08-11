# Fake News Detection Using Machine Learning

## 1. Introduction

Fake news can spread quickly and influence public opinion. This project
develops a supervised machine-learning system that classifies a news article
as fake or real from its written content. The project also provides a
Streamlit web interface for demonstrating predictions.

The system is a text-classification model, not an independent fact-checking
service. It learns patterns from the labels supplied by the dataset.

## 2. Problem Definition and Objective

**Problem type:** Binary classification.

**Input:** A news title and/or article body.

**Output:** `0 = fake news` or `1 = real news`.

The objectives are to:

1. Understand the dataset through EDA.
2. Clean and prepare the text data.
3. Convert text into numerical TF-IDF features.
4. Train and tune three approved algorithms.
5. Compare the models using classification metrics.
6. Deploy the prediction workflow with Streamlit.

## 3. Dataset Overview

The project uses the WELFake dataset. The original data contains 72,134
articles with title, text, and label fields. The labels are inherited from
the source dataset and are not independently fact-checked in this project.

After cleaning, 63,674 unique article contents remained:

- Fake news: 34,790 articles
- Real news: 28,884 articles
- Duplicate article content: 0
- Missing values in the modeling columns: 0

The data is reasonably balanced, although fake articles are more common than
real articles in the cleaned set.

## 4. Exploratory Data Analysis

The EDA stage inspected missing values, label distribution, duplicate content,
and article length. Missing title and text values were converted to empty
strings before the combined article content was created.

The generated evidence is stored in `outputs/figures/`:

- `class_distribution.png`
- `article_length_distribution.png`

The project also creates `outputs/reports/label_review_sample.csv` so that
sample inherited labels can be inspected transparently.

## 5. Data Cleaning and Preprocessing

The cleaning pipeline:

1. Loads the raw CSV.
2. Fills missing title and text values.
3. Normalizes label values to binary integers.
4. Combines title and body into a `content` field.
5. Normalizes repeated whitespace.
6. Removes duplicate article content.
7. Removes empty content.

The combined content is transformed using TF-IDF with unigrams and bigrams.
The vectorizer uses sublinear term frequency, a minimum document frequency of
2, and a maximum of 10,000 features.

The data is split into 80% training and 20% testing using stratification and a
fixed random seed of 42. For practical training time, model tuning uses a
reproducible stratified subset of 5,000 training articles. The final scores
are calculated on the untouched test set.

## 6. Machine-Learning Algorithms

### Random Forest

Random Forest combines many decision trees and averages their predictions.
It provides a useful ensemble baseline and can model nonlinear relationships.

### XGBoost

XGBoost builds trees sequentially, with later trees focusing on previous
errors. It is a strong gradient-boosting method for tabular feature data.

### LightGBM

LightGBM is an efficient gradient-boosting implementation designed for fast
training and large feature spaces. It achieved the strongest result in this
experiment.

All three models use the same TF-IDF features, split, test set, and evaluation
metrics to make the comparison fair.

## 7. Evaluation Metrics

- **Accuracy:** the proportion of all predictions that are correct.
- **Precision for real news:** among articles predicted as real, the proportion
  that are actually labeled real.
- **Recall for real news:** among articles labeled real, the proportion found
  by the model.
- **F1-score:** the harmonic mean of precision and recall.
- **ROC-AUC:** how well the model separates the two classes across thresholds.

F1-score and ROC-AUC are emphasized because accuracy alone does not show the
balance between precision and recall.

## 8. Results and Comparison

| Model | Accuracy | Precision (real) | Recall (real) | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.9017 | 0.8782 | 0.9095 | 0.8935 | 0.9698 |
| XGBoost | 0.9377 | 0.9122 | 0.9546 | 0.9329 | 0.9843 |
| LightGBM | **0.9482** | **0.9334** | 0.9538 | **0.9435** | **0.9892** |

LightGBM is the selected model because it achieved the highest accuracy,
F1-score, and ROC-AUC. XGBoost was the second-best model, while Random Forest
provided the baseline comparison.

The generated comparison charts are stored in `outputs/figures/`, including
the model comparison, ROC curves, and confusion matrices.

## 9. Streamlit Application

The Streamlit application provides:

- A text box for entering a news article.
- A selectable trained model.
- A fake/real prediction and confidence score.
- Prediction history.
- EDA charts and label filtering.
- Model comparison results.
- CSV download controls.

Run the application from the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app\app.py
```

The application uses port 8502.

## 10. Limitations

1. Dataset labels are inherited and may contain errors.
2. The model identifies language patterns; it does not verify facts.
3. TF-IDF does not fully understand context, sarcasm, or world knowledge.
4. The tuning search is intentionally small to fit a student laptop.
5. Performance on new sources or future events may differ from the test score.

## 11. Conclusion

This project completed the required workflow from data preparation through
deployment. Three approved algorithms were trained and compared fairly.
LightGBM was selected as the final model with an F1-score of 0.9435 and
ROC-AUC of 0.9892 on the held-out test set.

The Streamlit interface makes the model easy to demonstrate, while the
reported limitations clarify that the system is a predictive classifier and
not a replacement for professional fact-checking.

## 12. Future Improvements

Future work could include transformer-based language models, external
fact-checking sources, time-based validation, probability calibration, and a
larger manually verified evaluation set.
