# Presentation and Viva Guide

## Suggested slide structure

1. Title and team members
2. Problem statement and objective
3. Dataset overview and label definition
4. EDA and data-quality findings
5. Cleaning, TF-IDF, and train/test workflow
6. Random Forest, XGBoost, and LightGBM
7. Model comparison: metrics, confusion matrices, and ROC curves
8. Streamlit system demo
9. Challenges, limitations, and lessons learned
10. Conclusion and future work

## Images, icons, and layout

- Use a newspaper icon for the problem, a database icon for the dataset, a gear/pipeline icon for preprocessing, and a shield/check icon for the demo.
- Use the generated class-distribution, model-comparison, confusion-matrix, and ROC figures. Avoid decorative stock photos.
- Use a dark navy, white, and one accent color. Keep one message per slide, with a large chart or diagram on the right and three short points on the left.
- Use only a simple fade or appear animation when revealing a result. Avoid continuous, bouncing, or spinning animations.

## Likely lecturer questions and strong answers

- **Why did you choose Random Forest, XGBoost, and LightGBM?**  
  They are approved tree-based algorithms with different ensemble strategies. Comparing them with identical TF-IDF features makes the comparison fair.
- **Why is label 1 real news?**  
  That is the dataset definition. The cleaning pipeline preserves the mapping explicitly: 0 is fake and 1 is real.
- **Can you prove that every label is factually correct?**  
  No. The labels are inherited from the WELFake source. This project evaluates
  supervised classification against those dataset labels; it is not a manual
  fact-checking study. We provide a review sample so label quality can be
  discussed transparently.
- **Why remove duplicate articles?**  
  The same article in both training and testing can make the score unrealistically high, so duplicate combined content is removed before splitting.
- **Why use TF-IDF?**  
  It converts text into numbers and gives more weight to informative words while reducing the influence of common words.
- **Why split training and testing data?**  
  Training data is used to learn patterns; unseen test data measures generalization.
- **Why use F1-score instead of only accuracy?**  
  F1 balances precision and recall and is more informative when the cost of both types of error matters.
- **What is overfitting?**  
  Overfitting is memorizing training examples so the model performs poorly on new articles.
- **What is hyperparameter tuning?**  
  It tests model settings such as tree count, depth, learning rate, or number of leaves to find a better configuration.
- **What is Grid Search or Random Search?**  
  Grid Search tests every specified combination; Random Search samples a fixed number. This project uses a small randomized search to control training time.
- **Why did one algorithm perform better?**  
  Its tree-building strategy and selected settings fit the TF-IDF patterns better on the held-out test set. The measured result, not an assumption, determines the winner.

## Presentation tips

- Explain the workflow in order: data, cleaning, features, models, metrics, result, demo.
- Show one result per slide and define every metric before discussing it.
- State the limitation clearly: the model detects learned language patterns; it is not a fact-checking authority.
