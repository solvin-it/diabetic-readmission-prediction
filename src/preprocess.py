import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
from IPython.display import Markdown, display

def check_missing_values(dataframe):
    """
        This function checks for missing values in the given DataFrame and returns a summary of the missing values and their percentage.
        
        Parameters:
        dataframe (pd.DataFrame): The input DataFrame to check for missing values.
        
        Returns:
        pd.DataFrame: A DataFrame containing the count and percentage of missing values for each column.
    """

    df_copy = dataframe.copy()

    # Identify columns with missing values represented as '?' since the preview shows that some columns use '?' to indicate missing data
    columns_with_missing = df_copy.columns[df_copy.isin(['?']).any()]
    print(f"✓ Columns with missing values represented as '?': {list(columns_with_missing)}")

    df_copy.replace('?', np.nan, inplace=True)

    missing_values = df_copy.isnull().sum()
    missing_percentage = (missing_values / len(df_copy)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage
    })
    missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values(by='Missing Values', ascending=False)
    return missing_df

def check_for_outliers(dataframe, columns):
    """
        This function checks for outliers in the specified columns of the given DataFrame using the IQR method and returns a summary of the outliers.
        
        Parameters:
        dataframe (pd.DataFrame): The input DataFrame to check for outliers.
        columns (list): A list of column names to check for outliers.
        
        Returns:
        pd.DataFrame: A DataFrame containing the count and percentage of outliers in the specified columns.
    """

    outlier_summary = []
    for col in columns:
        Q1 = dataframe[col].quantile(0.25)
        Q3 = dataframe[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = dataframe[(dataframe[col] < lower_bound) | (dataframe[col] > upper_bound)]
        outlier_count = outliers.shape[0]
        outlier_percentage = (outlier_count / len(dataframe)) * 100
        outlier_summary.append({
            'Column': col,
            'Outlier Count': outlier_count,
            'Outlier Percentage': outlier_percentage
        })
    return pd.DataFrame(outlier_summary)

def display_univariate_analysis(dataframe, numerical_cols=None, categorical_cols=None, bins=30):
    """
        This function displays univariate analysis for numerical and categorical columns in the given DataFrame.
        
        Parameters:
        dataframe (pd.DataFrame): The input DataFrame to analyze.
        numerical_cols (list): A list of numerical column names to analyze.
        categorical_cols (list): A list of categorical column names to analyze.
    """

    def annotate_percentages(ax, total_count):
        for patch in ax.patches:
            height = patch.get_height()
            if height == 0:
                continue
            percentage = (height / total_count) * 100
            ax.annotate(
                f'{percentage:.1f}%',
                (patch.get_x() + patch.get_width() / 2, height),
                ha='center',
                va='bottom',
                xytext=(0, 4),
                textcoords='offset points'
            )

    if numerical_cols is None and categorical_cols is None:
        raise ValueError("At least one of numerical_cols or categorical_cols must be provided.")

    if numerical_cols is not None:
        print("✓ Univariate Analysis for Numerical Columns:")
        for col in numerical_cols:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Histogram
            sns.histplot(data=dataframe, x=col, bins=bins, kde=True, ax=axes[0])
            axes[0].set_title(f'Distribution of {col}')
            axes[0].set_xlabel(col)

            # Boxplot
            sns.boxplot(data=dataframe, x=col, ax=axes[1])
            axes[1].set_title(f'Boxplot of {col}')
            axes[1].set_xlabel(col)

            plt.tight_layout()
            plt.show()

        print("Summary statistics for numerical columns:")
        display(dataframe[numerical_cols].describe())

    if categorical_cols is not None:
        print("✓ Univariate Analysis for Categorical Columns:")
        for col in categorical_cols:
            non_null = dataframe[col].dropna()
            total = len(non_null)

            # If too many categories, show table instead
            if dataframe[col].nunique(dropna=True) > 10:
                print(f"✓ Showing value counts for {col} (more than 10 unique values):")
                counts = dataframe[col].value_counts(dropna=False)
                percentages = dataframe[col].value_counts(dropna=False, normalize=True) * 100

                summary_df = pd.DataFrame({
                    'count': counts,
                    'percentage (%)': percentages.round(1)
                })

                display(summary_df)
                continue

            plt.figure(figsize=(8, 4))
            ax = sns.countplot(data=dataframe, x=col, order=dataframe[col].value_counts().index)
            plt.title(f'Count Plot of {col}')
            plt.xlabel(col)
            plt.ylabel("Count")

            annotate_percentages(ax, total)

            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

def display_medication_eda(
    dataframe,
    medication_cols,
    target_col=None,
    positive_class=1,
    summary_top_n=8,
    analyze_target_top_n=8,
    analyze_target_min_active_use=None,
    category_order=("No", "Steady", "Up", "Down"),
    figsize=(10, 4)
):
    """
    Display notebook-friendly EDA for medication-related categorical features.

    This helper summarizes medication columns as a feature family first, then
    optionally performs bivariate analysis against the target only for the most
    informative medication variables based on active-use prevalence.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        Input dataframe containing medication columns and optional target column.

    medication_cols : list of str
        List of medication-related categorical column names.

    target_col : str, optional
        Target variable column name. If provided, the function performs bivariate
        target analysis for selected medication columns.

    positive_class : int or str, default=1
        Positive target class label used to compute positive-class rate and lift.

    summary_top_n : int, default=8
        Number of top medications by active use to highlight in the summary plots.

    analyze_target_top_n : int, default=8
        Number of medication columns to automatically select for target analysis
        based on active-use prevalence, unless analyze_target_min_active_use is used.

    analyze_target_min_active_use : float, optional
        Minimum active-use percentage required for a medication column to be included
        in target analysis. If provided, this takes priority over analyze_target_top_n.

    category_order : tuple, default=("No", "Steady", "Up", "Down")
        Expected category order for medication values.

    figsize : tuple, default=(10, 4)
        Figure size for plots.

    Returns:
    --------
    None
        Intended for notebook display only.
    """

    if not medication_cols:
        raise ValueError("medication_cols must contain at least one column name.")

    missing_cols = [col for col in medication_cols if col not in dataframe.columns]
    if missing_cols:
        raise ValueError(f"These medication columns are missing from the dataframe: {missing_cols}")

    if target_col is not None and target_col not in dataframe.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    display(Markdown("## Medication Feature Review"))
    display(Markdown(
        "This section summarizes medication-related variables as a feature family. "
        "The goal is to identify which medication columns have meaningful variation, "
        "which are highly sparse, and which deserve deeper analysis against the target."
    ))

    summary_rows = []

    for col in medication_cols:
        vc = dataframe[col].astype("string").value_counts(dropna=False, normalize=True) * 100
        unique_values = dataframe[col].astype("string").nunique(dropna=False)

        row = {
            "medication": col,
            "unique_values": unique_values,
            "pct_active_use": round(100 - vc.get("No", 0), 2)
        }

        for cat in category_order:
            row[f"pct_{cat.lower()}"] = round(vc.get(cat, 0), 2)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by="pct_active_use", ascending=False
    ).reset_index(drop=True)

    display(Markdown("### Medication Summary Table"))
    display(summary_df)

    display(Markdown(f"### Top {summary_top_n} Medications by Active Use"))
    display(summary_df.head(summary_top_n))

    # Plot active use ranking
    plt.figure(figsize=(max(figsize[0], 10), figsize[1]))
    sns.barplot(
        data=summary_df.head(summary_top_n),
        x="medication",
        y="pct_active_use"
    )
    plt.title(f"Top {summary_top_n} Medications by Active Use")
    plt.xlabel("Medication")
    plt.ylabel("Active Use (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Plot category breakdown for top medications
    plot_df = summary_df.head(summary_top_n).melt(
        id_vars=["medication", "unique_values", "pct_active_use"],
        value_vars=[f"pct_{cat.lower()}" for cat in category_order],
        var_name="status",
        value_name="percentage"
    )
    plot_df["status"] = plot_df["status"].str.replace("pct_", "", regex=False).str.title()

    plt.figure(figsize=(max(figsize[0], 10), figsize[1] + 1))
    sns.barplot(
        data=plot_df,
        x="medication",
        y="percentage",
        hue="status"
    )
    plt.title(f"Medication Status Distribution for Top {summary_top_n} Medications")
    plt.xlabel("Medication")
    plt.ylabel("Percentage")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # -----------------------------------------
    # Optional target-wise analysis
    # -----------------------------------------
    if target_col is not None:
        baseline_rate = (dataframe[target_col] == positive_class).mean() * 100

        # Auto-select columns for target analysis
        if analyze_target_min_active_use is not None:
            selected_cols = summary_df.loc[
                summary_df["pct_active_use"] >= analyze_target_min_active_use, "medication"
            ].tolist()
        else:
            selected_cols = summary_df["medication"].head(analyze_target_top_n).tolist()

        display(Markdown("## Medication Features vs Target"))
        display(Markdown(
            f"Overall positive class rate (`{target_col} = {positive_class}`): "
            f"**{baseline_rate:.2f}%**"
        ))
        display(Markdown(
            "Detailed target-wise analysis is restricted to medication variables with "
            "the greatest practical variation, based on active-use prevalence, plus any "
            "explicitly forced-in columns."
        ))

        selection_table = summary_df[summary_df["medication"].isin(selected_cols)].copy()
        display(Markdown("### Selected Medication Columns for Target Analysis"))
        display(selection_table.reset_index(drop=True))

        for col in selected_cols:
            display(Markdown(f"### {col}"))

            temp = dataframe[[col, target_col]].copy()
            temp[col] = temp[col].astype("string")

            count_table = pd.crosstab(temp[col], temp[target_col])
            row_pct = (pd.crosstab(temp[col], temp[target_col], normalize="index") * 100).round(2)

            display(Markdown("**Count Table**"))
            display(count_table)

            display(Markdown("**Row Percentage Table**"))
            display(row_pct)

            positive_rate = (
                temp.groupby(col)[target_col]
                .apply(lambda x: (x == positive_class).mean() * 100)
                .reset_index(name="positive_rate_pct")
            )

            positive_rate["lift_vs_baseline"] = (
                positive_rate["positive_rate_pct"] / baseline_rate
            ).round(3)

            positive_rate = positive_rate.sort_values("positive_rate_pct", ascending=False)

            display(Markdown("**Positive Class Rate and Lift vs Baseline**"))
            display(positive_rate.round(3))

            plt.figure(figsize=figsize)
            sns.barplot(
                data=positive_rate,
                x=col,
                y="positive_rate_pct"
            )
            plt.axhline(
                baseline_rate,
                color="red",
                linestyle="--",
                label=f"Baseline ({baseline_rate:.2f}%)"
            )
            plt.title(f"{col}: Positive Class Rate by Category")
            plt.xlabel(col)
            plt.ylabel(f"% with {target_col} = {positive_class}")
            plt.legend()
            plt.tight_layout()
            plt.show()

def display_bivariate_analysis(
    dataframe,
    target_col,
    numerical_cols=None,
    categorical_cols=None,
    positive_class=1,
    negative_class=0,
    top_n_categories=10,
    default_numeric_plot="box",
    show_fliers=False,
    zero_inflation_threshold=0.30,
    figsize=(8, 4)
):
    """
    Display notebook-friendly bivariate analysis between selected features and a target variable.

    This function is designed for exploratory data analysis inside a Jupyter notebook.
    It compares numerical and categorical features against a target variable and presents
    results in a format that is easier to interpret for preprocessing, feature engineering,
    and modeling decisions.

    For numerical features, the function:
    - computes class-wise mean and median
    - shows absolute and percentage differences between the target classes
    - visualizes each feature against the target
    - can automatically switch to histogram-based plots for zero-inflated variables,
    where standard boxplots are often hard to interpret

    For categorical features, the function:
    - displays count tables
    - displays row-normalized percentage tables
    - optionally groups rare categories into an "Other" bucket for readability
    - visualizes category-level differences across the target classes

    This helper is especially useful in structured/tabular machine learning projects where
    the goal is not only to visualize relationships, but also to identify practical next
    steps such as retaining, grouping, transforming, encoding, or deprioritizing features.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        Input dataframe containing the features and target column.

    target_col : str
        Name of the target variable column.

    numerical_cols : list of str, optional
        List of numerical columns to analyze against the target variable.
        If provided, the function will generate summary statistics and plots
        for each numerical feature.

    categorical_cols : list of str, optional
        List of categorical columns to analyze against the target variable.
        If provided, the function will generate contingency tables and plots
        for each categorical feature.

    positive_class : int, str, default=1
        Label representing the positive class in the target variable.
        Used when computing class differences for numerical summaries.

    negative_class : int, str, default=0
        Label representing the negative/reference class in the target variable.
        Used when computing class differences for numerical summaries.

    top_n_categories : int, default=10
        Maximum number of categories to display for high-cardinality categorical variables.
        Categories outside the top N most frequent values are grouped into "Other"
        for readability in tables and plots.

    default_numeric_plot : {'box', 'violin', 'hist'}, default='box'
        Default plot type to use for numerical variables unless automatic
        zero-inflation detection selects a histogram instead.

    show_fliers : bool, default=False
        Whether to display outlier points in boxplots.
        Setting this to False is often helpful when variables are highly skewed
        or contain extreme values that compress the box and whiskers.

    zero_inflation_threshold : float, default=0.30
        Threshold used to detect zero-inflated numerical variables.
        If the proportion of zero values in a numerical feature is greater than
        or equal to this threshold, the function may use a histogram instead of
        the default numeric plot, since boxplots are often less informative for
        heavily zero-inflated features.

    figsize : tuple, default=(8, 4)
        Figure size used for the generated plots.

    Returns:
    --------
    None
        This function is intended for notebook display and reporting.
        It primarily presents outputs directly in the notebook rather than
        returning processed objects for downstream use.

    Example:
    --------
    >>> display_bivariate_analysis(
    ...     dataframe=df_cleaned,
    ...     target_col='readmitted_binary',
    ...     numerical_cols=[
    ...         'time_in_hospital',
    ...         'num_medications',
    ...         'number_emergency',
    ...         'number_inpatient'
    ...     ],
    ...     default_numeric_plot='box',
    ...     show_fliers=False
    ... )

    >>> display_bivariate_analysis(
    ...     dataframe=df_cleaned,
    ...     target_col='readmitted_binary',
    ...     categorical_cols=[
    ...         'race',
    ...         'gender',
    ...         'age',
    ...         'A1Cresult',
    ...         'insulin',
    ...         'diabetesMed'
    ...     ],
    ...     top_n_categories=10
    ... )

    Notes:
    ------
    - This function is intended for exploratory analysis, not hypothesis testing.
    - Mean and median comparisons help reveal class separation, but they do not
    establish statistical significance on their own.
    - Percentage differences can appear very large when the baseline mean is close
    to zero, so they should always be interpreted together with absolute differences.
    - Zero-inflated healthcare utilization variables (for example, prior emergency
    or inpatient visits) often benefit from histogram-based visualization.
    - Grouping rare categorical levels into "Other" improves readability, but this
    grouping is intended for EDA/display purposes and should be reviewed carefully
    before applying the same strategy in final preprocessing.
    - If the target variable is highly imbalanced, visual differences should be
    interpreted alongside class proportions and later validated during modeling.
    """

    if numerical_cols is None and categorical_cols is None:
        raise ValueError("At least one of numerical_cols or categorical_cols must be provided.")

    # -----------------------------
    # NUMERICAL COLUMNS
    # -----------------------------
    if numerical_cols is not None:
        display(Markdown("## Bivariate Analysis: Numerical Features vs Target"))

        summary_rows = []

        for col in numerical_cols:
            display(Markdown(f"### {col}"))

            grouped = dataframe.groupby(target_col)[col].agg(["mean", "median"]).round(3)

            neg_mean = grouped.loc[negative_class, "mean"] if negative_class in grouped.index else np.nan
            pos_mean = grouped.loc[positive_class, "mean"] if positive_class in grouped.index else np.nan
            neg_median = grouped.loc[negative_class, "median"] if negative_class in grouped.index else np.nan
            pos_median = grouped.loc[positive_class, "median"] if positive_class in grouped.index else np.nan

            diff = pos_mean - neg_mean if pd.notnull(pos_mean) and pd.notnull(neg_mean) else np.nan
            pct_diff = ((diff / neg_mean) * 100) if pd.notnull(neg_mean) and neg_mean != 0 else np.nan

            zero_ratio = (dataframe[col] == 0).mean()

            # Auto-select plot type
            if zero_ratio >= zero_inflation_threshold:
                numeric_plot = "hist"
                display(Markdown(
                    f"**Auto plot selection:** `{col}` is zero-inflated "
                    f"({zero_ratio:.1%} zeros), so a histogram is used."
                ))
            else:
                numeric_plot = default_numeric_plot

            detail_table = pd.DataFrame({
                negative_class: [neg_mean, neg_median],
                positive_class: [pos_mean, pos_median]
            }, index=["mean", "median"])

            display(detail_table)

            summary_rows.append({
                "feature": col,
                f"mean_{negative_class}": round(neg_mean, 3),
                f"mean_{positive_class}": round(pos_mean, 3),
                f"median_{negative_class}": round(neg_median, 3),
                f"median_{positive_class}": round(pos_median, 3),
                "diff": round(diff, 3) if pd.notnull(diff) else np.nan,
                "pct_diff": round(pct_diff, 3) if pd.notnull(pct_diff) else np.nan,
                "zero_ratio": round(zero_ratio * 100, 2)
            })

            plt.figure(figsize=figsize)

            if numeric_plot == "box":
                sns.boxplot(
                    data=dataframe,
                    x=target_col,
                    y=col,
                    showfliers=show_fliers
                )

            elif numeric_plot == "violin":
                sns.violinplot(
                    data=dataframe,
                    x=target_col,
                    y=col,
                    cut=0
                )

            elif numeric_plot == "hist":
                sns.histplot(
                    data=dataframe,
                    x=col,
                    hue=target_col,
                    kde=False,
                    stat="density",
                    common_norm=False,
                    element="step"
                )

            else:
                raise ValueError("default_numeric_plot must be 'box', 'violin', or 'hist'")

            plt.title(f"{col} by {target_col}")
            plt.xlabel(col if numeric_plot == "hist" else target_col)
            plt.ylabel("Density" if numeric_plot == "hist" else col)
            plt.tight_layout()
            plt.show()

        summary_df = pd.DataFrame(summary_rows).set_index("feature")

        display(Markdown("## Numerical Summary Table"))
        display(summary_df)

    # -----------------------------
    # CATEGORICAL COLUMNS
    # -----------------------------
    if categorical_cols is not None:
        display(Markdown("## Bivariate Analysis: Categorical Features vs Target"))

        for col in categorical_cols:
            display(Markdown(f"### {col}"))

            temp_df = dataframe[[col, target_col]].copy()

            # Convert to string for display/grouping safety
            temp_df[col] = temp_df[col].astype("string")

            value_counts = temp_df[col].value_counts(dropna=False)

            if len(value_counts) > top_n_categories:
                top_categories = value_counts.head(top_n_categories).index
                temp_df[col] = temp_df[col].where(temp_df[col].isin(top_categories), other="Other")

            count_table = pd.crosstab(temp_df[col], temp_df[target_col])
            pct_table = (pd.crosstab(temp_df[col], temp_df[target_col], normalize="index") * 100).round(2)

            display(Markdown("**Count Table**"))
            display(count_table)

            display(Markdown("**Row Percentage Table**"))
            display(pct_table)

            plot_df = pct_table.reset_index().melt(
                id_vars=col,
                var_name=target_col,
                value_name="percentage"
            )

            plt.figure(figsize=(max(figsize[0], 10), figsize[1] + 1))
            sns.barplot(
                data=plot_df,
                x=col,
                y="percentage",
                hue=target_col
            )
            plt.title(f"{col} vs {target_col} (Row %)")
            plt.xlabel(col)
            plt.ylabel("Percentage")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()

def display_correlation_analysis(
    dataframe,
    numerical_cols,
    target_col=None,
    method="pearson",
    top_n_pairs=10,
    feature_figsize=(10, 8),
    target_figsize=(6, 4)
):
    """
    Display notebook-friendly correlation analysis for numerical features,
    including feature-to-target and feature-to-feature relationships.

    This helper is intended for exploratory data analysis inside a notebook.
    It presents correlation outputs in a way that is useful for preprocessing,
    feature selection, redundancy checks, and interpretation before modeling.

    If a target column is provided and is numeric (for example, a binary target
    encoded as 0 and 1), the function also computes and visualizes correlation
    between each numerical feature and the target.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        Input dataframe containing the numerical features and optional target.

    numerical_cols : list of str
        Numerical columns to include in the correlation analysis.

    target_col : str, optional
        Target variable column name. If provided, the function computes feature-
        to-target correlation in addition to feature-to-feature correlation.

    method : str, default='pearson'
        Correlation method to use. Common choices are:
        - 'pearson' for linear correlation
        - 'spearman' for monotonic/rank-based correlation

    top_n_pairs : int, default=10
        Number of strongest feature-to-feature correlation pairs to display.

    feature_figsize : tuple, default=(10, 8)
        Figure size for the numerical feature correlation heatmap.

    target_figsize : tuple, default=(6, 4)
        Figure size for the feature-to-target correlation heatmap.

    Returns:
    --------
    None
        This function is intended for notebook display and reporting rather than
        returning analysis objects.

    Notes:
    ------
    - Feature-to-feature correlation is mainly used to detect redundancy and
      possible multicollinearity among numerical predictors.
    - Feature-to-target correlation is useful for identifying direct linear
      association, but low correlation does not imply a feature is unimportant.
      Some variables may still be useful through nonlinear effects or interactions.
    - In binary classification, Pearson correlation with a 0/1 target is valid
      as a quick screening tool, but it should not be treated as a final feature
      selection criterion.
    """

    # ----------------------------------------
    # Validation
    # ----------------------------------------
    if numerical_cols is None or len(numerical_cols) == 0:
        raise ValueError("numerical_cols must contain at least one numerical column.")

    missing_cols = [col for col in numerical_cols if col not in dataframe.columns]
    if missing_cols:
        raise ValueError(f"These numerical columns are missing from the dataframe: {missing_cols}")

    if target_col is not None and target_col not in dataframe.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    # ----------------------------------------
    # Part 1: Correlation with Target
    # ----------------------------------------
    if target_col is not None:
        display(Markdown("### Numerical Feature Correlation with Target"))

        target_corr = (
            dataframe[list(numerical_cols) + [target_col]]
            .corr(method=method)[[target_col]]
            .drop(index=target_col)
            .sort_values(by=target_col, ascending=False)
        )

        display(Markdown("**Feature-to-Target Correlation Table**"))
        display(target_corr.round(3))

        plt.figure(figsize=target_figsize)
        sns.heatmap(
            target_corr,
            annot=True,
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            cbar_kws={"label": f"{method.title()} Correlation"}
        )
        plt.title("Numerical Feature Correlation with Target")
        plt.tight_layout()
        plt.show()

        top_positive = target_corr[target_corr[target_col] > 0].head(3)
        top_negative = target_corr[target_corr[target_col] < 0].head(3)

        display(Markdown("**Interpretation Guide**"))
        display(Markdown(
            "- Positive values indicate that higher feature values tend to be associated "
            "with the positive target class.\n"
            "- Negative values indicate an inverse relationship with the target.\n"
            "- Correlations close to zero suggest weak linear association, though such "
            "features may still be useful in nonlinear or interaction-based models."
        ))

        if not top_positive.empty:
            display(Markdown("**Top Positive Correlations with Target**"))
            display(top_positive.round(3))

        if not top_negative.empty:
            display(Markdown("**Top Negative Correlations with Target**"))
            display(top_negative.round(3))

    # ----------------------------------------
    # Part 2: Feature-to-Feature Correlation
    # ----------------------------------------
    display(Markdown("### Numerical Feature Correlation Analysis"))

    corr_matrix = dataframe[numerical_cols].corr(method=method)

    display(Markdown("**Correlation Matrix**"))
    display(corr_matrix.round(3))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=feature_figsize)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8, "label": f"{method.title()} Correlation"}
    )
    plt.title("Correlation Heatmap of Numerical Features", pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # ----------------------------------------
    # Part 3: Strongest Pairwise Correlations
    # ----------------------------------------
    corr_pairs = (
        corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool))
        .stack()
        .reset_index()
    )
    corr_pairs.columns = ["feature_1", "feature_2", "correlation"]

    corr_pairs["pair_key"] = corr_pairs.apply(
        lambda row: tuple(sorted([row["feature_1"], row["feature_2"]])),
        axis=1
    )
    corr_pairs = corr_pairs.drop_duplicates(subset="pair_key").drop(columns="pair_key")

    corr_pairs["abs_correlation"] = corr_pairs["correlation"].abs()
    corr_pairs = corr_pairs.sort_values("abs_correlation", ascending=False)

    display(Markdown(f"**Top {top_n_pairs} Strongest Pairwise Correlations**"))
    display(corr_pairs.head(top_n_pairs).reset_index(drop=True).round(3))

    # ----------------------------------------
    # Part 4: Interpretation Notes
    # ----------------------------------------
    display(Markdown("### Interpretation Notes"))
    display(Markdown(
        "- **Feature-to-feature correlation** helps identify potentially redundant "
        "variables and possible multicollinearity concerns, especially for linear models.\n"
        "- **Feature-to-target correlation** is useful for screening direct linear "
        "signal, but it should not be treated as a final measure of feature importance.\n"
        "- Tree-based models can still benefit from features with weak target correlation "
        "if those features contribute through nonlinear splits or interactions.\n"
        "- If very strong pairwise correlations are found, consider whether both features "
        "should be retained, transformed, regularized, or reviewed for redundancy."
    ))

def add_utilization_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer binary flags and log1p transforms for the zero-inflated utilization columns.

    All three prior-utilization variables (number_outpatient, number_emergency,
    number_inpatient) are heavily zero-inflated with long right tails. This function
    addresses both issues simultaneously:

    Binary flags ('had_outpatient', 'had_emergency', 'had_inpatient', 'had_procedures')
        Capture any-vs-none variation cleanly. For highly zero-inflated columns the
        binary flag often carries as much discriminative signal as the raw count and is
        robust to extreme values.

    Log1p transforms ('log_outpatient', 'log_emergency', 'log_inpatient')
        Compress the right tail for linear and distance-based models. log1p(x) = log(1+x)
        is safe for zero values (log1p(0) = 0) while substantially reducing skewness.

    The original count columns are retained so that tree-based models (Random Forest,
    XGBoost) can partition on raw values without any information loss.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame after leakage filtering.

    Returns
    -------
    pd.DataFrame
        DataFrame with utilization flags and log1p transforms added as new columns.
    """
    df_out = dataframe.copy()

    util_cols  = ['number_outpatient', 'number_emergency', 'number_inpatient']
    flag_names = ['had_outpatient',    'had_emergency',    'had_inpatient']
    log_names  = ['log_outpatient',    'log_emergency',    'log_inpatient']

    for raw, flag, log in zip(util_cols, flag_names, log_names):
        df_out[flag] = (df_out[raw] > 0).astype(int)
        df_out[log]  = np.log1p(df_out[raw])

    df_out['had_procedures'] = (df_out['num_procedures'] > 0).astype(int)

    print("✓ Binary utilization flags added:", flag_names + ['had_procedures'])
    print("✓ Log1p transforms added:", log_names)
    return df_out

def recode_admission_type(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate rare/unknown admission_type_id codes into a single 'Unknown' category.

    The EDA identified that codes 5, 6, and 8 represent:
      - 5: Not Available
      - 6: NULL
      - 8: Not Mapped
    
    These ~10% of records represent administratively undefined admission types.
    Consolidating them into a single 'Unknown' category prevents the model from
    learning spurious patterns on rare administrative codes.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame after gender cleaning.

    Returns
    -------
    pd.DataFrame
        DataFrame with admission codes 5,6,8 consolidated to 'Unknown'.
    """
    df_out = dataframe.copy()
    unknown_admission_codes = {5, 6, 8}
    
    # Map unknown codes to the string 'Unknown'
    df_out['admission_type_id'] = df_out['admission_type_id'].apply(
        lambda x: 'Unknown' if x in unknown_admission_codes else x
    )
    
    n_unknown = (dataframe['admission_type_id'].isin(unknown_admission_codes)).sum()
    print(f"✓ Consolidated {n_unknown:,} records ({100*n_unknown/len(dataframe):.2f}%) with admission codes {{5,6,8}} → 'Unknown'")
    print(f"  Admission type value counts:")
    print(df_out['admission_type_id'].value_counts().to_string())
    return df_out

def group_discharge_disposition(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse discharge_disposition_id into four clinically coherent string groups.

    The 25+ distinct numeric codes are replaced with interpretable labels that
    reflect the patient's post-discharge care trajectory:

        'home'      Discharged to home (with or without home-health support).
                    Raw codes: 1, 6, 8
                    Baseline group; lowest mean readmission rate (~9.3%).

        'facility'  Transferred to a post-acute or long-term care facility
                    (SNF, ICF, rehab unit, long-term hospital, federal facility,
                    psychiatric hospital, CAH, etc.).
                    Raw codes: 2, 3, 4, 5, 10, 15, 22, 23, 24, 25, 26, 27, 28, 29, 30
                    Elevated readmission risk; includes the highest-risk SNF/rehab codes.

        'inpatient' Still in care or expected to return for inpatient services.
                    Raw codes: 9, 12

        'other'     AMA (left against medical advice), hospice discharge, outpatient
                    referrals, unknown, unmapped, and any remaining codes not assigned
                    to the three groups above.

    Note: discharge_disposition_id == 11 (Expired) is excluded upstream by
    filter_leakage_records(), so it will not appear here.

    The original numeric column is replaced in-place with the string labels.
    One-hot encoding is deferred to the feature-engineering step.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame after utilization feature engineering.

    Returns
    -------
    pd.DataFrame
        DataFrame with discharge_disposition_id replaced by string group labels.
    """
    df_out = dataframe.copy()

    discharge_map = {
        1: 'home', 6: 'home', 8: 'home',
        2: 'facility', 3: 'facility',  4: 'facility',  5: 'facility',
        10: 'facility', 15: 'facility', 22: 'facility', 23: 'facility',
        24: 'facility', 25: 'facility', 26: 'facility', 27: 'facility',
        28: 'facility', 29: 'facility', 30: 'facility',
        9: 'inpatient', 12: 'inpatient',
    }

    df_out['discharge_disposition_id'] = (
        df_out['discharge_disposition_id']
        .map(discharge_map)
        .fillna('other')
    )

    print("✓ discharge_disposition_id grouped into clinical categories:")
    print(df_out['discharge_disposition_id'].value_counts().to_string())
    return df_out

def group_admission_source(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse admission_source_id into four clinically interpretable string groups.

    The coded numeric values are mapped to interpretable labels that capture the
    clinical meaning of how a patient arrived at the hospital:

        'emergency_room'  Arrived directly via the emergency department.
                          Raw codes: 7
                          Highest-volume group; likely higher acuity on arrival.

        'referral'        Admitted via a physician or clinic referral, representing
                          a planned or semi-planned admission.
                          Raw codes: 1, 2

        'transfer'        Transferred from another healthcare facility (hospital, SNF,
                          HMO, ambulatory surgery center, hospice, or other care setting).
                          Raw codes: 3, 4, 5, 6, 10, 18, 22, 25, 26
                          Bivariate analysis showed HMO referral (code 3, 15.51%) and
                          transfer codes carried elevated readmission rates, motivating
                          a dedicated group rather than folding into 'other'.

        'other'           Court/Law Enforcement, not-available, null, unknown,
                          and any remaining codes not assigned above.

    The original numeric column is replaced with the string labels.
    One-hot encoding is deferred to the feature-engineering step.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame after discharge disposition grouping.

    Returns
    -------
    pd.DataFrame
        DataFrame with admission_source_id replaced by string group labels.
    """
    df_out = dataframe.copy()

    admission_map = {
        7:  'emergency_room',
        1:  'referral', 2: 'referral',
        3:  'transfer', 4: 'transfer', 5:  'transfer', 6: 'transfer',
        10: 'transfer', 18: 'transfer', 22: 'transfer', 25: 'transfer', 26: 'transfer',
    }

    df_out['admission_source_id'] = (
        df_out['admission_source_id']
        .map(admission_map)
        .fillna('other')
    )

    print("✓ admission_source_id grouped into clinical categories:")
    print(df_out['admission_source_id'].value_counts().to_string())
    return df_out

def encode_clinical_flags(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary indicator flags from categorical clinical variables and recode
    change/diabetesMed to 0/1 integers.

    New columns created:

        specialty_known   1 if medical_specialty is not 'Unknown', else 0.
                          Separates the informative specialty signal from pure
                          missingness; allows models to use both dimensions of
                          this feature without conflating them.

        insulin_adjusted  1 if insulin was actively changed during admission
                          ('Down' or 'Up'), else 0. Captures whether the
                          treating physician adjusted dosage — a signal of
                          suboptimal glycaemic control on entry.

        glucose_tested    1 if max_glu_serum is not 'none', else 0.
                          Whether an inpatient glucose test was ordered is a
                          clinical decision that may independently distinguish
                          high-acuity from routine admissions.

        A1C_tested        1 if A1Cresult is not 'none', else 0.
                          Analogous reasoning to glucose_tested.

    Columns recoded in-place:

        change      'Ch' → 1 (medication change occurred), 'No' → 0.
        diabetesMed 'Yes' → 1 (diabetes medication prescribed), 'No' → 0.

    The original insulin, max_glu_serum, and A1Cresult columns are retained
    with their level categories for use in feature engineering.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame after admission source grouping.

    Returns
    -------
    pd.DataFrame
        DataFrame with the four indicator columns added and change/diabetesMed recoded.
    """
    df_out = dataframe.copy()

    df_out['specialty_known']  = (df_out['medical_specialty'] != 'Unknown').astype(int)
    df_out['insulin_adjusted'] = df_out['insulin'].isin(['Down', 'Up']).astype(int)
    df_out['glucose_tested']   = (df_out['max_glu_serum'] != 'none').astype(int)
    df_out['A1C_tested']       = (df_out['A1Cresult'] != 'none').astype(int)

    df_out['change']      = df_out['change'].map({'Ch': 1, 'No': 0})
    df_out['diabetesMed'] = df_out['diabetesMed'].map({'Yes': 1, 'No': 0})

    print("✓ Binary indicator flags added: specialty_known, insulin_adjusted, glucose_tested, A1C_tested")
    print("✓ Recoded: change (Ch→1 / No→0), diabetesMed (Yes→1 / No→0)")
    return df_out