import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegressionCV, LassoLarsIC, ElasticNetCV
from scipy.stats import norm
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

def prep_feat_df(eval_results, preference_results, args, eval_axes):
    preference_results["score"] = preference_results["avg_final_scores"]
    preference_results["axis"] = "preference"
    eval_results["score"] = eval_results["avg_final_scores"]

    df_pref = pd.concat([eval_results, preference_results])
    feat_df = create_feature_df(df_pref, args).dropna()
    
    feat_df1 = feat_df.copy()
    feat_df2 = feat_df.copy()
    feat_df1["model_label"] = 0
    feat_df2["model_label"] = 1
    for axis in eval_axes:
        feat_df2[axis] = -feat_df2[axis]
    feat_df2["preference"] = -feat_df2["preference"]
    
    return feat_df, pd.concat([feat_df1, feat_df2])

def train_and_evaluate(train_features, test_features, eval_axes, label_column, args, tag, iteration, model_type="logistic"):
    X_train = train_features[eval_axes]
    y_train = train_features[label_column].apply(lambda x: 1 if x > 0 else 0)

    X_test = test_features[eval_axes]
    y_test = test_features[label_column].apply(lambda x: 1 if x > 0 else 0)

    # Train model and unpack the returned values
    (
        accuracy,
        feature_importance,
        test_results,
        train_accuracy,
        train_results,
        summary,
        feature_order,
    ) = train_model(X_train, y_train, X_test, y_test, eval_axes, label_column, model_type=model_type, x_val=(tag == "train"))

    # Update results
    train_results["question"] = train_features["question"]
    test_results["question"] = test_features["question"]

    for model in args.models:
        train_results[model] = train_features[model]
        test_results[model] = test_features[model]

    return {
        "accuracy": accuracy,
        "feature_importance": feature_importance,
        "test_results": test_results,
        "train_accuracy": train_accuracy,
        "feature_order": feature_order,
    }

def compute_p_values(model, X_train, y_train):
    # Wald test for logistic regression
    n = X_train.shape[0]
    p = X_train.shape[1]

    coefs = np.concatenate([model.intercept_, model.coef_.flatten()])
    predictions = model.predict_proba(X_train)

    # Compute the matrix of weights for each observation
    # w = p * (1 - p) for logistic regression
    p_hat = predictions[:, 1]
    W = np.diag(p_hat * (1 - p_hat))

    X_train_with_intercept = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    cov_matrix = np.linalg.inv(
        np.dot(np.dot(X_train_with_intercept.T, W), X_train_with_intercept)
    )
    standard_errors = np.sqrt(np.diag(cov_matrix))
    t_stats = coefs / standard_errors
    p_values = [2 * (1 - stats.t.cdf(np.abs(t), df=n - p - 1)) for t in t_stats]

    return np.array(p_values[1:]), np.array(t_stats[1:])


def get_lars_feature_order(X, y):
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit LARS model
    lars_model = LassoLarsIC(criterion="aic")
    lars_model.fit(X_scaled, y)

    # Get the order of features based on the absolute values of coefficients
    coef_abs = np.abs(lars_model.coef_)
    feature_order = [X.columns[i] for i in np.argsort(coef_abs)[::-1]]

    return feature_order


def train_single_model(
    X_train, y_train, X_test=None, y_test=None, model_type="logistic", x_val=False
):
    # Get LARS feature order
    lars_feature_order = get_lars_feature_order(X_train, y_train)

    if model_type in ["ridge", "lasso"]:
        if model_type == "ridge":
            model = LogisticRegressionCV(
                cv=5, penalty="l2", solver="lbfgs", max_iter=1000
            )
        else:  # lasso
            model = LogisticRegressionCV(
                cv=5, penalty="l1", solver="saga", max_iter=1000
            )

        model.fit(X_train, y_train)

        if x_val:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            accuracies = cross_val_score(
                model, X_train, y_train, cv=cv, scoring="accuracy"
            )
            accuracy = np.mean(accuracies)
        else:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

        feature_analysis = pd.DataFrame(
            {
                "feature": X_train.columns,
                "coefficient": model.coef_[0],
                "lars_order": [
                    lars_feature_order.index(feat) for feat in X_train.columns
                ],
            }
        )
        feature_analysis["odds_ratio"] = np.exp(feature_analysis["coefficient"])
        feature_analysis = feature_analysis.sort_values("lars_order")

        results = model
    else:  # logistic regression
        # this is a new model that I've been trying out to better understand feature importance, not used in the paper 
        model = LogisticRegressionCV(
            penalty="elasticnet", solver="saga", l1_ratios=[0.5]
        )
        # model = LogisticRegressionCV()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Compute p-values
        p_values, t_stats = compute_p_values(model, X_train, y_train)
        feature_analysis = pd.DataFrame(
            {
                "feature": X_train.columns,
                "coefficient": model.coef_[0],
                "p_value": p_values,
                "t_stat": t_stats,
                "lars_order": [
                    lars_feature_order.index(feat) for feat in X_train.columns
                ],
            }
        )
        feature_analysis["odds_ratio"] = np.exp(feature_analysis["coefficient"])
        feature_analysis = feature_analysis.sort_values("lars_order")

        results = model

    return accuracy, feature_analysis, results, lars_feature_order


def train_model(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_columns,
    label_column,
    model_type="logistic",
    x_val=False,
):
    # Train the model
    accuracy, feature_analysis, results, lars_feature_order = train_single_model(
        X_train, y_train, X_test, y_test, model_type=model_type, x_val=x_val
    )

    # Prepare test results
    test_df = X_test.copy()
    test_df[label_column] = y_test
    test_df["predicted"] = results.predict(X_test)
    test_df["correct"] = test_df["predicted"] == test_df[label_column]

    # Prepare train results
    train_pred = results.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    train_df = X_train.copy()
    train_df[label_column] = y_train
    train_df["predicted"] = train_pred
    train_df["correct"] = train_df["predicted"] == train_df[label_column]

    summary = f"{model_type.capitalize()} Regression Results:\nBest C: {results.C_[0]}\nAccuracy: {accuracy}"

    return (
        accuracy,
        feature_analysis,
        test_df,
        train_accuracy,
        train_df,
        summary,
        lars_feature_order,
    )


import matplotlib.pyplot as plt
import seaborn as sns


def get_score(item):
    if float(item[0]) == float(item[1]):
        return 0
    if float(item[0]) > 0 and float(item[1]) < 0:
        return 1
    if float(item[0]) < 0 and float(item[1]) > 0:
        return -1
    return 0  # Default case


def create_feature_df(df, args):
    axes = df["axis"].unique()
    features = []
    for axis in axes:
        axis_df = df[df["axis"] == axis]
        axis_df = axis_df[["question"] + list(args.models) + ["score"]]
        axis_df.columns = ["question"] + list(args.models) + [axis]
        features.append(axis_df)
    # Merge on models columns to get a single dataframe
    feature_df = features[0]
    for i in range(1, len(features)):
        feature_df = pd.merge(
            feature_df,
            features[i],
            on=["question"] + list(args.models),
        )
    return feature_df


def calculate_vibe_overlap(df, vibe1, vibe2, models, plot=False):
    """
    Calculate the overlap between two vibes based on their scores across samples.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the scores for each sample and axis.
                           Should have columns 'question', 'axis', and 'score'.
    vibe1 (str): Name of the first vibe.
    vibe2 (str): Name of the second vibe.
    plot (bool): If True, generate a scatter plot of the two vibes' scores.

    Returns:
    float: Pearson correlation coefficient between the two vibes' scores.
    """
    # Create a pivot table with questions as rows and vibes as columns
    pivot_df = df[df["axis"].isin([vibe1, vibe2])].pivot(
        index=["question", *models], columns="axis", values="score"
    )

    # Calculate the Pearson correlation coefficient
    correlation, _ = stats.pearsonr(pivot_df[vibe1], pivot_df[vibe2])

    if plot:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=vibe1, y=vibe2, data=pivot_df)
        plt.title(f"Correlation between {vibe1} and {vibe2}: {correlation:.2f}")
        plt.xlabel(f"{vibe1} scores")
        plt.ylabel(f"{vibe2} scores")
        plt.show()

    return correlation


def generate_vibe_overlap_heatmap(df, vibes, models):
    """
    Generate a heatmap of correlations between all pairs of vibes.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the scores for each sample and axis.
                           Should have columns 'question', 'axis', and 'score'.
    vibes (list): List of vibe names to include in the heatmap.

    Returns:
    pandas.DataFrame: Correlation matrix of vibe overlaps.
    """
    pivot_df = df[df["axis"].isin(vibes)].pivot(
        index=["question", *models], columns="axis", values="score"
    )
    correlation_matrix = pivot_df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0
    )
    plt.title("Vibe Overlap Heatmap")
    # pil image pof the heatmap
    plt.savefig("heatmap.png", bbox_inches="tight")

    return correlation_matrix, "heatmap.png"


def calculate_multi_vibe_separability_score(df, axes, models):
    """
    Calculate the multi-vibe separability score for a given set of axes.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the scores for each sample and axis.
                           Should have 'question' as index and axes as columns.
    axes (list): List of axis names to include in the calculation.

    Returns:
    tuple: (raw_score, normalized_score)
    """
    # Create a pivot table with questions as rows and axes as columns
    pivot_df = df[df["axis"].isin(axes)].pivot(
        index=["question", *models], columns="axis", values="score"
    )

    # Ensure we only use the specified axes
    pivot_df = pivot_df[axes]

    # Calculate the L2 norm (Euclidean distance) for each sample
    l2_norms = np.sqrt((pivot_df**2).sum(axis=1))

    # Calculate the raw multi-vibe separability score
    raw_score = l2_norms.mean()

    # Calculate the normalized score
    k = len(axes)
    normalized_score = raw_score / np.sqrt(k)

    return raw_score, normalized_score


from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# This was something I played around with to remove vibes which have little impact on the model but decided not to use it
def permutation_importance_filtering(X, y, n_repeats=1000, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )

    model = LogisticRegression(random_state=random_state)
    model.fit(X_train, y_train)

    result = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, random_state=random_state
    )

    # Filter vibes: importance > 0 AND lower bound of confidence interval > 0
    # important_vibes = np.where((result.importances_mean > 0) &
    #                            (result.importances_mean - 2*result.importances_std > 0))[0]
    important_vibes = np.where((result.importances_mean > 0))[0]

    return {
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
        "is_important": [
            True if i in important_vibes else False
            for i in range(len(result.importances_mean))
        ],
    }
