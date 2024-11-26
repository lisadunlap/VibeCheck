import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import plot_tree
from sklearn.utils import shuffle

def expand_dataframe_with_axes(df, metric='avg_diff_scores'):
  unique_axes = df['axis'].unique()
  existing_pairs = set(zip(df['question'], df['axis']))
  new_rows = []
  for question in df['question'].unique():
      for axis in unique_axes:
          if (question, axis) not in existing_pairs:
              new_row = {
                  'question': question,
                  'axis': axis,
                  'avg_scores': [0, 0],
                  metric: [0, 0]
              }
              new_rows.append(new_row)
  new_rows_df = pd.DataFrame(new_rows)
  return pd.concat([df, new_rows_df], ignore_index=True)

def prepare_data_for_decision_tree(df, models, metric='avg_diff_scores'):
  df = df.copy()
  results_short = df[['question', 'topic', 'axis', 'avg_scores', metric]]
  df = expand_dataframe_with_axes(results_short)

  # Create separate rows for each model in the models list
  expanded_rows = []
  for i, model in enumerate(models):
      model_rows = df.copy()
      model_rows['label'] = model
      model_rows[metric] = model_rows[metric].apply(lambda x: x[i])  # Select the element for the current model
      expanded_rows.append(model_rows)

  # Concatenate all model rows
  expanded_df = pd.concat(expanded_rows, ignore_index=True)

  # Pivot the data to create feature columns for each axis
  pivot_df = expanded_df.pivot_table(index=['question', 'label'], columns='axis', values=metric, fill_value=0).reset_index()

  # Prepare features and target
  X = pivot_df.drop(columns=['question', 'label'])
  y = pivot_df['label']

  return X, y

def train_decision_tree(results, test_results, models, show=False, metric='avg_diff_scores'):
  X_train, y_train = prepare_data_for_decision_tree(results, models, metric=metric)
  X_test, y_test = prepare_data_for_decision_tree(test_results, models, metric=metric)
  X_train, y_train = shuffle(X_train, y_train, random_state=42)
  data = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
  # shuffle=True is the default
  clf = DecisionTreeClassifier(random_state=42)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  y_train_pred = clf.predict(X_train)

  print(f"Training a decsion tree with {len(models)} models on {len(X_train)} samples")
  # print("Train Classification Report:")
  # print(classification_report(y_train, y_train_pred))
  print("Test Classification Report:")
  print(classification_report(y_test, y_pred))
  # list feature importances
  print("Feature Importances:")
  for i, f in enumerate(X_train.columns):
    print(f"{f}: {clf.feature_importances_[i]}")

  data["y_pred_tree"] = y_pred
  data["feature_importances_tree"] = clf.feature_importances_

  if show:
    fig = plt.figure(figsize=(40,25))
    plot_tree(clf, 
      filled=True, 
      feature_names=[c.replace("High", "\nHigh").replace("Low", "\nLow") for c in X_train.columns], 
      class_names=models, 
      fontsize=12,  # Set the font size
      proportion=True,  # Set nodes to be proportional to the number of samples
      rounded=True  # Round the nodes
      )
    plt.show()

  clf_reg = LogisticRegression(random_state=42)
  clf_reg.fit(X_train, y_train)
  y_pred_reg = clf_reg.predict(X_test)
  y_train_pred = clf_reg.predict(X_train)

  print(f"Training a logistic regression with {len(models)} models on {len(X_train)} samples")
  # print("Train Classification Report:")
  # print(classification_report(y_train, y_train_pred))
  print("Test Classification Report:")
  print(classification_report(y_test, y_pred_reg))
  # list feature importances
  print("Feature Importances:")
  for i, f in enumerate(X_train.columns):
    print(f"{f}: {clf_reg.coef_[0][i]}")

  data["y_pred_reg"] = y_pred_reg
  data["feature_importances_reg"] = clf_reg.coef_[0]

  return clf, clf_reg, data

def train_individual_feature_impact(results, test_results, models, metric='avg_diff_scores'):
    X_train, y_train = prepare_data_for_decision_tree(results, models, metric=metric)
    X_test, y_test = prepare_data_for_decision_tree(test_results, models, metric=metric)
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    
    feature_impact = {}
    logistic_reg_feature_impact = {}
    
    for feature in X_train.columns:
        
        # Train decision tree on individual feature
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train[[feature]], y_train)
        y_pred = clf.predict(X_test[[feature]])
        feature_impact[feature] = classification_report(y_test, y_pred, output_dict=True)

        # Train logistic regression on individual feature
        clf_reg = LogisticRegression(random_state=42)
        clf_reg.fit(X_train[[feature]], y_train)
        y_pred_reg = clf_reg.predict(X_test[[feature]])
        logistic_reg_feature_impact[feature] = classification_report(y_test, y_pred_reg, output_dict=True)

    # get overall acc using all features
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    feature_impact["all_features"] = classification_report(y_test, y_pred, output_dict=True)

    clf_reg = LogisticRegression(random_state=42)
    clf_reg.fit(X_train, y_train)
    y_pred_reg = clf_reg.predict(X_test)
    logistic_reg_feature_impact["all_features"] = classification_report(y_test, y_pred_reg, output_dict=True)
    
    return clf, clf_reg, {"decision_tree": feature_impact, "logistic_regression": logistic_reg_feature_impact}