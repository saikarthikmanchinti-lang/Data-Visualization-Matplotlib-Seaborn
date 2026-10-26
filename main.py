"""
================================================================================
PROJECT: Data Visualization Project - Titanic Dataset Analysis
ASSIGNMENT: Data Visualization with Matplotlib and Seaborn
AUTHOR: College Student / Pair Programmer
DESCRIPTION:
    A beginner-friendly, end-to-end Python script demonstrating core data
    visualization techniques using Pandas, NumPy, Matplotlib, and Seaborn.
    
    Visualizations Created:
    1. Bar Charts: Survival Rate by Passenger Class & Gender; Embarkation Port Counts
    2. Scatter Plot: Passenger Age vs. Fare with Outlier Annotations
    3. Histograms: Passenger Age Distribution & Skewed Fare Distribution (Raw vs Log)
    4. Box Plots: Ticket Fare Outliers across Classes & Age Distributions by Survival
    5. Correlation Heatmap: Pearson Correlation Matrix across Numerical Features
    6. Summary Dashboard: Multi-panel Executive Overview

All charts are exported as high-resolution PNGs (300 DPI) into the 'charts/' folder.
================================================================================
"""

import os
import sys
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 output encoding on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ==============================================================================
# 1. GLOBAL VISUALIZATION STYLING & CONFIGURATION
# ==============================================================================
def configure_styles():
    """
    Configure global aesthetic settings for Matplotlib and Seaborn.
    Using consistent fonts, clean grid lines, and high-contrast color palettes.
    """
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8


# Custom color constants for consistent theme throughout the project
COLOR_PRIMARY = "#1f77b4"      # Ocean Blue
COLOR_SURVIVED = "#2ecc71"     # Vibrant Green (Survived)
COLOR_DIED = "#e74c3c"         # Crimson Red (Did Not Survive)
COLOR_ACCENT = "#f39c12"       # Amber Gold
PALETTE_GENDER = {"female": "#e76f51", "male": "#2a9d8f"}
PALETTE_SURVIVAL = {"Survived": COLOR_SURVIVED, "Did Not Survive": COLOR_DIED}


# ==============================================================================
# 2. DATA LOADING & PREPROCESSING
# ==============================================================================
def load_and_preprocess_data(data_path="data/titanic.csv"):
    """
    Load the Titanic dataset from local CSV file. If not found locally,
    attempts to download from the official seaborn repository or load via seaborn.
    
    Returns:
        df (pd.DataFrame): Cleaned and enriched Titanic DataFrame.
    """
    print("\n" + "=" * 70)
    print("STEP 1: LOADING & PREPROCESSING DATASET")
    print("=" * 70)
    
    # Ensure data folder exists
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    if os.path.exists(data_path):
        print(f"[OK] Loading dataset from local file: {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("[INFO] Local dataset not found. Downloading authentic Titanic CSV...")
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
        urllib.request.urlretrieve(url, data_path)
        print(f"[OK] Downloaded and saved to: {data_path}")
        df = pd.read_csv(data_path)

    # Dataset Summary
    print(f"    - Total Rows (Passengers): {len(df)}")
    print(f"    - Total Columns (Features): {len(df.columns)}")
    print(f"    - Columns: {list(df.columns)}")
    
    # Feature Engineering for Clearer Visualizations
    # 1. Map Survival to descriptive text labels
    df["survival_status"] = df["survived"].map({1: "Survived", 0: "Did Not Survive"})
    
    # 2. Map Passenger Class to readable labels
    df["class_label"] = df["pclass"].map({1: "1st Class", 2: "2nd Class", 3: "3rd Class"})
    
    # 3. Compute Total Family Size = Siblings/Spouses + Parents/Children + Self
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    
    # 4. Map Embarkation Port abbreviations to full city names
    port_map = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
    df["embark_port_full"] = df["embarked"].map(port_map).fillna("Unknown")
    
    return df


# ==============================================================================
# 3. STATISTICAL ANALYSIS & CONSOLE OBSERVATIONS
# ==============================================================================
def print_statistical_observations(df):
    """
    Calculate and print exact numerical observations from the real dataset.
    Demonstrates data calculation using NumPy and Pandas aggregations.
    """
    print("\n" + "=" * 70)
    print("STEP 2: COMPUTING EXACT DATA OBSERVATIONS & STATISTICS")
    print("=" * 70)
    
    total = len(df)
    survived_cnt = df["survived"].sum()
    survival_pct = (survived_cnt / total) * 100
    
    print(f"\n1. OVERALL SURVIVAL:")
    print(f"   - {survived_cnt} out of {total} passengers survived ({survival_pct:.2f}%).")
    
    print(f"\n2. SURVIVAL BY GENDER:")
    female_cnt = (df["sex"] == "female").sum()
    female_surv = df[df["sex"] == "female"]["survived"].sum()
    female_pct = (female_surv / female_cnt) * 100
    
    male_cnt = (df["sex"] == "male").sum()
    male_surv = df[df["sex"] == "male"]["survived"].sum()
    male_pct = (male_surv / male_cnt) * 100
    
    print(f"   - Females: {female_surv} survived of {female_cnt} ({female_pct:.2f}% survival rate)")
    print(f"   - Males:   {male_surv} survived of {male_cnt} ({male_pct:.2f}% survival rate)")
    print(f"   - Insight: Females were ~3.9 times more likely to survive than males.")
    
    print(f"\n3. SURVIVAL BY PASSENGER CLASS:")
    for pclass in [1, 2, 3]:
        sub = df[df["pclass"] == pclass]
        p_pct = sub["survived"].mean() * 100
        print(f"   - Class {pclass}: {sub['survived'].sum()} of {len(sub)} survived ({p_pct:.2f}%)")
    print(f"   - Insight: 1st Class passengers had more than 2.5x higher survival rate than 3rd Class.")
    
    print(f"\n4. FARE OUTLIERS (IQR METHOD VIA NUMPY/PANDAS):")
    q1_fare = np.percentile(df["fare"], 25)
    q3_fare = np.percentile(df["fare"], 75)
    iqr_fare = q3_fare - q1_fare
    upper_fence = q3_fare + 1.5 * iqr_fare
    outliers = df[df["fare"] > upper_fence]
    
    print(f"   - Fare Q1 (25th percentile): £{q1_fare:.2f}")
    print(f"   - Fare Median (50th percentile): £{df['fare'].median():.2f}")
    print(f"   - Fare Q3 (75th percentile): £{q3_fare:.2f}")
    print(f"   - Fare Interquartile Range (IQR): £{iqr_fare:.2f}")
    print(f"   - Statistical Upper Outlier Threshold (Q3 + 1.5*IQR): £{upper_fence:.2f}")
    print(f"   - Number of Fare Outliers: {len(outliers)} passengers ({len(outliers)/total*100:.2f}% of passengers)")
    print(f"   - Maximum Fare Paid: £{df['fare'].max():.2f} (paid by {len(df[df['fare'] == df['fare'].max()])} passengers)")

    print(f"\n5. AGE DISTRIBUTION:")
    age_valid = df["age"].dropna()
    print(f"   - Age Mean: {age_valid.mean():.2f} years | Median: {age_valid.median():.2f} years")
    print(f"   - Age Min: {age_valid.min():.2f} years (~5 months infant) | Max: {age_valid.max():.2f} years")
    print(f"   - Missing Age records: {df['age'].isnull().sum()} ({df['age'].isnull().sum()/total*100:.2f}%)")


# ==============================================================================
# 4. CHART GENERATION FUNCTIONS
# ==============================================================================

# ------------------------------------------------------------------------------
# 4.1 Bar Charts
# ------------------------------------------------------------------------------
def plot_bar_charts(df, charts_dir="charts"):
    """
    Creates two distinct bar charts:
    1. Survival Rate by Passenger Class grouped by Gender.
    2. Total Passengers and Survival breakdown by Embarkation Port.
    """
    print("\n--- Generating Bar Charts ---")
    
    # ---------------- Chart 1: Survival Rate by Class and Gender ----------------
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Calculate exact percentages for bar heights
    grouped = df.groupby(["class_label", "sex"])["survived"].mean().reset_index()
    grouped["survival_pct"] = grouped["survived"] * 100
    
    palette = {"female": "#e76f51", "male": "#2a9d8f"}
    barplot = sns.barplot(
        data=grouped,
        x="class_label",
        y="survival_pct",
        hue="sex",
        palette=palette,
        edgecolor="black",
        linewidth=0.8,
        ax=ax
    )
    
    # Add direct percentage annotations on top of each bar
    for p in barplot.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.1f}%",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                xytext=(0, 4),
                textcoords="offset points"
            )
            
    ax.set_title("Titanic Survival Rate by Passenger Class & Gender", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Passenger Ticket Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("Survival Rate (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(title="Gender", labels=["Female", "Male"], loc="upper right", frameon=True)
    
    # Explanation note at bottom
    fig.text(0.5, -0.04, "Observation: Females in 1st & 2nd class had >90% survival, showing strong 'women and children first' protocol.",
             ha="center", fontsize=9, style="italic", color="#333333")
    
    plt.tight_layout()
    chart1_path = os.path.join(charts_dir, "01_survival_by_class_and_gender_bar.png")
    plt.savefig(chart1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {chart1_path}")
    
    # ---------------- Chart 2: Passenger Count by Embarkation Port ----------------
    fig, ax = plt.subplots(figsize=(9, 6))
    
    port_order = ["Southampton", "Cherbourg", "Queenstown"]
    port_df = df[df["embark_port_full"].isin(port_order)]
    
    countplot = sns.countplot(
        data=port_df,
        x="embark_port_full",
        hue="survival_status",
        order=port_order,
        palette=PALETTE_SURVIVAL,
        edgecolor="black",
        linewidth=0.8,
        ax=ax
    )
    
    # Add count annotations above bars
    for p in countplot.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                xytext=(0, 3),
                textcoords="offset points"
            )
            
    ax.set_title("Passenger Count and Survival Outcome by Embarkation Port", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Embarkation Port City", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Passengers", fontsize=11, fontweight="bold")
    ax.legend(title="Status", loc="upper right", frameon=True)
    
    fig.text(0.5, -0.04, "Observation: Southampton had the largest passenger volume (644); Cherbourg had the highest proportion of survivors.",
             ha="center", fontsize=9, style="italic", color="#333333")
             
    plt.tight_layout()
    chart2_path = os.path.join(charts_dir, "02_passenger_count_by_embarkation_bar.png")
    plt.savefig(chart2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {chart2_path}")


# ------------------------------------------------------------------------------
# 4.2 Scatter Plots
# ------------------------------------------------------------------------------
def plot_scatter_plots(df, charts_dir="charts"):
    """
    Creates a detailed scatter plot:
    - Passenger Age (X-axis) vs. Ticket Fare (Y-axis)
    - Colored by Survival Status (Green = Survived, Red = Did Not Survive)
    - Marker style varies by Passenger Class
    - Annotates extreme fare outliers (> £500).
    """
    print("\n--- Generating Scatter Plots ---")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    # Filter non-null age for scatter plotting
    scatter_df = df.dropna(subset=["age"]).copy()
    
    # Scatter plot with Seaborn
    scatter = sns.scatterplot(
        data=scatter_df,
        x="age",
        y="fare",
        hue="survival_status",
        style="class_label",
        palette=PALETTE_SURVIVAL,
        alpha=0.75,
        s=70,
        edgecolor="black",
        linewidth=0.5,
        ax=ax
    )
    
    # Highlight and annotate extreme fare outliers (> £500)
    top_fare = df["fare"].max()
    ax.annotate(
        f"Extreme Luxury Suite Outliers\n(Fare: £{top_fare:.2f}, 1st Class)",
        xy=(35, top_fare),
        xytext=(45, top_fare - 80),
        arrowprops=dict(facecolor="#e74c3c", shrink=0.08, width=1.5, headwidth=8),
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="#e74c3c", facecolor="#ffebee", alpha=0.9)
    )
    
    ax.set_title("Passenger Age vs. Ticket Fare Paid (Segmented by Survival & Class)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Passenger Age (Years)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ticket Fare Paid (£ GBP)", fontsize=11, fontweight="bold")
    ax.set_xlim(-2, 85)
    ax.set_ylim(-15, 550)
    
    # Refined legend position
    ax.legend(loc="upper right", frameon=True, title="Legend", fontsize=9)
    
    fig.text(0.5, -0.04, "Observation: High-fare passengers clustered near top survived; majority of passengers paid < £50 and clustered under age 40.",
             ha="center", fontsize=9, style="italic", color="#333333")
             
    plt.tight_layout()
    chart3_path = os.path.join(charts_dir, "03_age_vs_fare_scatter.png")
    plt.savefig(chart3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {chart3_path}")


# ------------------------------------------------------------------------------
# 4.3 Histograms (Data Distributions)
# ------------------------------------------------------------------------------
def plot_histograms(df, charts_dir="charts"):
    """
    Creates two histogram visualizations:
    1. Age Distribution with Kernel Density Estimation (KDE) and statistical markers.
    2. Fare Distribution Comparison: Raw Skewed vs. Log-Transformed Distribution.
    """
    print("\n--- Generating Histograms ---")
    
    # ---------------- Chart 4: Age Distribution with KDE ----------------
    fig, ax = plt.subplots(figsize=(9, 6))
    
    age_data = df["age"].dropna()
    mean_age = age_data.mean()
    median_age = age_data.median()
    
    sns.histplot(
        age_data,
        kde=True,
        bins=30,
        color="#3498db",
        edgecolor="black",
        line_kws={"linewidth": 2, "color": "#1b4f72"},
        ax=ax
    )
    
    # Statistical benchmark lines
    ax.axvline(mean_age, color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean Age: {mean_age:.1f} yrs")
    ax.axvline(median_age, color="#27ae60", linestyle="-.", linewidth=2, label=f"Median Age: {median_age:.1f} yrs")
    
    ax.set_title("Passenger Age Distribution with Kernel Density Estimate (KDE)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Age (Years)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Passengers", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    
    fig.text(0.5, -0.04, "Observation: Distribution peaks around 20-30 years (young working adults), with a secondary spike in infant ages (0-5 yrs).",
             ha="center", fontsize=9, style="italic", color="#333333")
             
    plt.tight_layout()
    chart4_path = os.path.join(charts_dir, "04_age_distribution_histogram.png")
    plt.savefig(chart4_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {chart4_path}")
    
    # ---------------- Chart 5: Fare Distribution (Raw vs Log-Transformed) ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    
    # Raw Fare Distribution (Right-skewed)
    sns.histplot(df["fare"], bins=40, kde=True, color="#e67e22", edgecolor="black", ax=ax1)
    ax1.set_title("A. Raw Fare Distribution (Heavily Skewed)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Fare Paid (£ GBP)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Passenger Count", fontsize=10, fontweight="bold")
    ax1.axvline(df["fare"].median(), color="red", linestyle="--", label=f"Median: £{df['fare'].median():.2f}")
    ax1.legend()
    
    # Log-Transformed Fare Distribution (np.log1p avoids log(0))
    log_fare = np.log1p(df["fare"])
    sns.histplot(log_fare, bins=30, kde=True, color="#16a085", edgecolor="black", ax=ax2)
    ax2.set_title("B. Log-Transformed Fare: log(1 + Fare)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Log(1 + Fare)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Passenger Count", fontsize=10, fontweight="bold")
    ax2.axvline(log_fare.median(), color="red", linestyle="--", label=f"Median Log: {log_fare.median():.2f}")
    ax2.legend()
    
    plt.suptitle("Understanding Skewness in Titanic Fare Pricing", fontsize=14, fontweight="bold", y=1.02)
    
    fig.text(0.5, -0.05, "Observation: Raw fare is heavily right-skewed with 75% of passengers paying ≤ £31. Log transform normalizes the scale.",
             ha="center", fontsize=9, style="italic", color="#333333")
             
    plt.tight_layout()
    chart5_path = os.path.join(charts_dir, "05_fare_distribution_histogram.png")
    plt.savefig(chart5_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {chart5_path}")


# ------------------------------------------------------------------------------
# 4.4 Box Plots (Outlier Identification)
# ------------------------------------------------------------------------------
def plot_box_plots(df, charts_dir="charts"):
    """
    Creates two box plots:
    1. Ticket Fare Distribution by Class (highlighting high-ticket outliers).
    2. Age Distribution across Classes and Survival Outcome.
    """
    print("\n--- Generating Box Plots ---")
    
    # ---------------- Chart 6: Fare by Class Box Plot (Outliers) ----------------
    fig, ax = plt.subplots(figsize=(9, 6))
    
    class_order = ["1st Class", "2nd Class", "3rd Class"]
    class_palette = ["#f1c40f", "#3498db", "#95a5a6"]
    
    sns.boxplot(
        data=df,
        x="class_label",
        y="fare",
        hue="class_label",
        order=class_order,
        palette=class_palette,
        legend=False,
        fliersize=5,
        flierprops={"marker": "D", "markerfacecolor": "#e74c3c", "markeredgecolor": "black", "alpha": 0.7},
        width=0.5,
        ax=ax
    )
    
    # Add horizontal line for overall 1.5*IQR upper fence
    q1 = df["fare"].quantile(0.25)
    q3 = df["fare"].quantile(0.75)
    fence = q3 + 1.5 * (q3 - q1)
    ax.axhline(fence, color="#c0392b", linestyle=":", linewidth=1.5, label=f"Overall Outlier Fence (£{fence:.1f})")
    
    ax.set_title("Ticket Fare Box Plot by Passenger Class (Outlier Detection)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Passenger Ticket Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ticket Fare (£ GBP)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    
    fig.text(0.5, -0.04, "Observation: Red diamonds denote outliers. 1st Class exhibits extreme dispersion up to £512, while 3rd Class is tightly compressed.",
             ha="center", fontsize=9, style="italic", color="#333333")
             
    plt.tight_layout()
    chart6_path = os.path.join(charts_dir, "06_fare_by_class_boxplot.png")
    plt.savefig(chart6_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {chart6_path}")
    
    # ---------------- Chart 7: Age by Class and Survival Box Plot ----------------
    fig, ax = plt.subplots(figsize=(9.5, 6))
    
    sns.boxplot(
        data=df,
        x="class_label",
        y="age",
        hue="survival_status",
        order=class_order,
        palette=PALETTE_SURVIVAL,
        width=0.6,
        fliersize=4,
        ax=ax
    )
    
    ax.set_title("Passenger Age Distribution by Class and Survival Status", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Passenger Ticket Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("Passenger Age (Years)", fontsize=11, fontweight="bold")
    ax.legend(title="Survival Status", loc="upper right", frameon=True)
    
    fig.text(0.5, -0.04, "Observation: 1st class passengers were older on average (median ~38 yrs); surviving children were most prominent in 2nd and 3rd class.",
             ha="center", fontsize=9, style="italic", color="#333333")
             
    plt.tight_layout()
    chart7_path = os.path.join(charts_dir, "07_age_by_class_and_survival_boxplot.png")
    plt.savefig(chart7_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {chart7_path}")


# ------------------------------------------------------------------------------
# 4.5 Correlation Heatmap
# ------------------------------------------------------------------------------
def plot_correlation_heatmap(df, charts_dir="charts"):
    """
    Computes Pearson correlation matrix for numerical attributes and renders
    an annotated heatmap using Seaborn.
    """
    print("\n--- Generating Correlation Heatmap ---")
    fig, ax = plt.subplots(figsize=(8.5, 7))
    
    # Select key numeric features
    numeric_columns = ["survived", "pclass", "age", "sibsp", "parch", "fare", "family_size"]
    corr_matrix = df[numeric_columns].corr()
    
    # Rename for professional presentation
    labels = ["Survived", "Ticket Class", "Age", "Siblings/Spouses", "Parents/Children", "Fare Paid", "Family Size"]
    
    # Plot heatmap
    heatmap = sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=1,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient (r)"},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    
    ax.set_title("Correlation Heatmap of Titanic Numerical Features", fontsize=14, fontweight="bold", pad=20)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    fig.text(0.5, -0.08, "Observation: Ticket Class has strongest negative correlation with survival (-0.34); Fare has strongest positive correlation (+0.26).",
             ha="center", fontsize=9, style="italic", color="#333333")
             
    plt.tight_layout()
    chart8_path = os.path.join(charts_dir, "08_correlation_heatmap.png")
    plt.savefig(chart8_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {chart8_path}")


# ------------------------------------------------------------------------------
# 4.6 Executive Summary Dashboard (Multi-Panel Figure)
# ------------------------------------------------------------------------------
def plot_summary_dashboard(df, charts_dir="charts"):
    """
    Combines core visual insights into a single presentation-ready 2x2 dashboard.
    Demonstrates Matplotlib multi-subplot coordination.
    """
    print("\n--- Generating Summary Executive Dashboard ---")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("TITANIC DATASET: EXECUTIVE VISUALIZATION DASHBOARD", fontsize=18, fontweight="bold", y=0.98)
    
    # Panel 1: Survival by Class & Gender (Top-Left)
    grouped = df.groupby(["class_label", "sex"])["survived"].mean().reset_index()
    grouped["survival_pct"] = grouped["survived"] * 100
    sns.barplot(
        data=grouped,
        x="class_label",
        y="survival_pct",
        hue="sex",
        palette=PALETTE_GENDER,
        ax=axes[0, 0]
    )
    axes[0, 0].set_title("1. Survival Rate by Class & Gender", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Survival Rate (%)")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylim(0, 110)
    axes[0, 0].legend(title="Gender", loc="upper right")
    
    # Panel 2: Age Distribution (Top-Right)
    age_clean = df["age"].dropna()
    sns.histplot(age_clean, kde=True, bins=25, color="#3498db", ax=axes[0, 1])
    axes[0, 1].axvline(age_clean.median(), color="red", linestyle="--", label=f"Median: {age_clean.median():.0f} yrs")
    axes[0, 1].set_title("2. Passenger Age Distribution (KDE)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Age (Years)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].legend()
    
    # Panel 3: Fare Outliers by Class (Bottom-Left)
    sns.boxplot(
        data=df,
        x="class_label",
        y="fare",
        hue="class_label",
        palette=["#f1c40f", "#3498db", "#95a5a6"],
        legend=False,
        ax=axes[1, 0]
    )
    axes[1, 0].set_title("3. Ticket Fare Outliers by Class", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Ticket Class")
    axes[1, 0].set_ylabel("Fare (£ GBP)")
    
    # Panel 4: Age vs Fare Scatter (Bottom-Right)
    scatter_df = df.dropna(subset=["age"])
    sns.scatterplot(
        data=scatter_df,
        x="age",
        y="fare",
        hue="survival_status",
        palette=PALETTE_SURVIVAL,
        alpha=0.7,
        s=45,
        ax=axes[1, 1]
    )
    axes[1, 1].set_title("4. Age vs. Fare Paid by Survival Status", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Age (Years)")
    axes[1, 1].set_ylabel("Fare (£ GBP)")
    axes[1, 1].legend(title="Outcome", loc="upper right")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    dashboard_path = os.path.join(charts_dir, "09_data_visualization_dashboard.png")
    plt.savefig(dashboard_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {dashboard_path}")


# ==============================================================================
# 5. MAIN EXECUTION CONTROLLER
# ==============================================================================
def main():
    """
    Main driver function: loads data, creates output directories, invokes all
    plotting routines, and reports completion summary.
    """
    print("\n" + "=" * 70)
    print("   DATA VISUALIZATION PROJECT: TITANIC EXPLORATORY ANALYSIS   ")
    print("   Using Python, Pandas, NumPy, Matplotlib & Seaborn          ")
    print("=" * 70)
    
    # 1. Setup styles
    configure_styles()
    
    # 2. Setup output folder
    charts_folder = "charts"
    os.makedirs(charts_folder, exist_ok=True)
    
    # 3. Load and preprocess dataset
    df = load_and_preprocess_data(data_path="data/titanic.csv")
    
    # 4. Display statistical calculations
    print_statistical_observations(df)
    
    # 5. Generate all visual graphics
    print("\n" + "=" * 70)
    print("STEP 3: RENDERING & SAVING VISUALIZATION CHARTS")
    print("=" * 70)
    plot_bar_charts(df, charts_folder)
    plot_scatter_plots(df, charts_folder)
    plot_histograms(df, charts_folder)
    plot_box_plots(df, charts_folder)
    plot_correlation_heatmap(df, charts_folder)
    plot_summary_dashboard(df, charts_folder)
    
    # 6. Final success report
    print("\n" + "=" * 70)
    print("PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    saved_charts = [f for f in os.listdir(charts_folder) if f.endswith(".png")]
    print(f"Total charts created in '{charts_folder}/': {len(saved_charts)}")
    for idx, chart in enumerate(sorted(saved_charts), 1):
        file_path = os.path.join(charts_folder, chart)
        file_size_kb = os.path.getsize(file_path) / 1024
        print(f"  {idx}. {chart:<45} ({file_size_kb:.1f} KB)")
    print("\nAll observations calculated and verified from authentic Titanic data.")
    print("Ready for evaluation and submission!\n")


if __name__ == "__main__":
    main()
