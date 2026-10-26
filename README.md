# Data_Visualization_Project: Titanic Dataset Analysis
> **College Assignment:** Data Visualization with Matplotlib and Seaborn  
> **Course:** Data Science / Python Programming  
> **Tools:** Python 3, Pandas, NumPy, Matplotlib, Seaborn  

---

## 📋 Table of Contents
1. [Project Objective](#-project-objective)
2. [Dataset Overview](#-dataset-overview)
3. [Project Structure](#-project-structure)
4. [Installation & Requirements](#-installation--requirements)
5. [How to Run the Project](#-how-to-run-the-project)
6. [Calculated Data Observations & Visual Insights](#-calculated-data-observations--visual-insights)
   - [1. Bar Charts: Survival by Class & Gender](#1-bar-chart-survival-rate-by-passenger-class--gender)
   - [2. Bar Charts: Passenger Volume by Embarkation Port](#2-bar-chart-passenger-volume--survival-by-embarkation-port)
   - [3. Scatter Plot: Age vs. Fare Paid with Outlier Annotation](#3-scatter-plot-passenger-age-vs-ticket-fare-paid)
   - [4. Histogram: Passenger Age Distribution with KDE](#4-histogram-passenger-age-distribution-with-kde)
   - [5. Histogram: Raw vs. Log-Transformed Fare Distribution](#5-histogram-fare-distribution-skewness-analysis)
   - [6. Box Plot: Fare Distribution & Outlier Identification by Class](#6-box-plot-ticket-fare-by-class-outlier-detection)
   - [7. Box Plot: Age Distribution by Class and Survival](#7-box-plot-passenger-age-by-class-and-survival-status)
   - [8. Correlation Heatmap: Pearson Correlation Matrix](#8-correlation-heatmap-numerical-feature-correlations)
   - [9. Executive Dashboard: Multi-Panel Summary Overview](#9-executive-dashboard-multi-panel-summary)
7. [Visualization Techniques & Best Practices](#-visualization-techniques--best-practices)
8. [Conclusion & Key Findings](#-conclusion--key-findings)

---

## 🎯 Project Objective
The primary objective of this project is to apply exploratory data visualization principles to uncover patterns, socio-demographic trends, and statistical outliers within the historical Titanic passenger manifest.

Key educational goals:
- Utilize **Pandas** for structured data extraction, missing-value profiling, and grouped aggregations.
- Utilize **NumPy** for numerical computations (percentiles, IQR thresholds, and logarithmic transformations).
- Utilize **Matplotlib** for canvas architecture, multi-subplot configuration, fine-tuned annotations, axis controls, and high-resolution exports.
- Utilize **Seaborn** for high-level statistical visualization including Kernel Density Estimation (KDE), categorical box plots, count plots, and correlation heatmaps.
- Export presentation-ready graphics (300 DPI PNGs) to a dedicated `charts/` folder with accompanying empirical observations calculated from actual data.

---

## 📊 Dataset Overview
The project analyzes the real **Titanic passenger dataset** (891 passenger records across 15 attributes).

| Column Name | Data Type | Description | Sample Values / Range |
| :--- | :--- | :--- | :--- |
| `survived` | Integer | Survival indicator (0 = No, 1 = Yes) | 0, 1 |
| `pclass` | Integer | Ticket Class (1 = 1st, 2 = 2nd, 3 = 3rd) | 1, 2, 3 |
| `sex` | Object | Biological gender of passenger | `female`, `male` |
| `age` | Float | Age in years (177 missing values) | 0.42 to 80.00 years |
| `sibsp` | Integer | Number of siblings or spouses aboard | 0 to 8 |
| `parch` | Integer | Number of parents or children aboard | 0 to 6 |
| `fare` | Float | Passenger ticket fare in British Pounds (£) | £0.00 to £512.33 |
| `embarked` | Object | Port of Embarkation (C, Q, S) | `S`, `C`, `Q` |
| `class` | Category | Categorical class name | `First`, `Second`, `Third` |
| `who` | Object | Demographic persona | `man`, `woman`, `child` |
| `adult_male`| Boolean | True if passenger is an adult male | `True`, `False` |
| `deck` | Category | Cabin deck assigned (688 missing) | `A`, `B`, `C`, `D`, `E`, `F`, `G` |
| `embark_town`| Object| Full name of embarkation port town | `Southampton`, `Cherbourg`, `Queenstown` |
| `alive` | Object | Survival outcome text | `yes`, `no` |
| `alone` | Boolean | True if passenger traveled without family | `True`, `False` |

### Preprocessing & Derived Features Added:
- `survival_status`: Readable string labels (`"Survived"`, `"Did Not Survive"`).
- `class_label`: Formatted passenger class names (`"1st Class"`, `"2nd Class"`, `"3rd Class"`).
- `family_size`: Computed as `sibsp + parch + 1`.
- `embark_port_full`: Mapped port codes to full city names (`"Southampton"`, `"Cherbourg"`, `"Queenstown"`).

---

## 📁 Project Structure

```
Data_Visualization_Project/
│
├── charts/                                        # Generated high-resolution visualization charts (300 DPI)
│   ├── 01_survival_by_class_and_gender_bar.png
│   ├── 02_passenger_count_by_embarkation_bar.png
│   ├── 03_age_vs_fare_scatter.png
│   ├── 04_age_distribution_histogram.png
│   ├── 05_fare_distribution_histogram.png
│   ├── 06_fare_by_class_boxplot.png
│   ├── 07_age_by_class_and_survival_boxplot.png
│   ├── 08_correlation_heatmap.png
│   └── 09_data_visualization_dashboard.png
│
├── data/                                          # Dataset storage folder
│   └── titanic.csv                                # Authentic Titanic dataset (891 rows, 15 columns)
│
├── main.py                                        # Complete, documented Python visualization script
├── requirements.txt                               # Required Python dependencies
└── README.md                                      # Comprehensive assignment report and documentation
```

---

## ⚙️ Installation & Requirements

### Prerequisites
- Python 3.9 or higher (tested and verified on Python 3.14)
- `pip` package manager

### Required Libraries
All dependencies are specified in [requirements.txt](file:///c:/Users/pavan/OneDrive/Desktop/INT%202/requirements.txt):
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `matplotlib>=3.7.0`
- `seaborn>=0.12.0`

### Setup Instructions
Open your terminal or command prompt in the project root directory and run:

```bash
# 1. (Optional) Create and activate a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate

# 2. Install required packages
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

Execute the main script from your terminal:

```bash
python main.py
```

### What Happens When You Run `main.py`?
1. **Data Ingestion**: Loads `data/titanic.csv` (or automatically downloads it if missing).
2. **Statistical Calculation**: Calculates exact survival percentages, IQR metrics, age statistics, and correlation coefficients from the live data.
3. **Chart Rendering**: Generates all 9 charts using Matplotlib and Seaborn with custom color palettes, titles, labels, legends, and direct annotations.
4. **Export**: Saves every figure into the `charts/` folder at 300 DPI resolution.
5. **Execution Report**: Displays a detailed progress and summary report in the console.

---

## 📈 Calculated Data Observations & Visual Insights

All observations below are calculated directly from the authentic 891 passenger records.

### 1. Bar Chart: Survival Rate by Passenger Class & Gender
- **File:** `charts/01_survival_by_class_and_gender_bar.png`
- **Plot Type:** Grouped Bar Chart with direct percentage labels.
- **Empirical Findings:**
  - **1st Class Females:** **96.81% survived** (91 out of 94 survived).
  - **2nd Class Females:** **92.11% survived** (70 out of 76 survived).
  - **3rd Class Females:** **50.00% survived** (72 out of 144 survived).
  - **1st Class Males:** **36.89% survived** (45 out of 122 survived).
  - **2nd Class Males:** **15.74% survived** (17 out of 108 survived).
  - **3rd Class Males:** **13.54% survived** (47 out of 347 survived).
- **Analytical Takeaway:**
  Gender was the single strongest determinant of survival: females overall experienced a **74.20% survival rate** versus **18.89% for males** (~3.9 times higher). However, class stratification remained severe: a female in 3rd class had roughly even odds of survival (50.0%), whereas nearly all 1st and 2nd class women were saved (>92%).

---

### 2. Bar Chart: Passenger Volume & Survival by Embarkation Port
- **File:** `charts/02_passenger_count_by_embarkation_bar.png`
- **Plot Type:** Categorical Count Bar Chart with survival breakdown.
- **Empirical Findings:**
  - **Southampton (S):** 644 passengers boarded (72.4% of total). 217 survived (33.70% survival rate), 427 perished.
  - **Cherbourg (C):** 168 passengers boarded (18.9% of total). 93 survived (55.36% survival rate), 75 perished.
  - **Queenstown (Q):** 77 passengers boarded (8.7% of total). 30 survived (38.96% survival rate), 47 perished.
- **Analytical Takeaway:**
  Cherbourg was the only embarkation port where survivors outnumbered casualties (55.36% survival). Cross-referencing reveals this was because a disproportionately large share of Cherbourg passengers booked expensive 1st class cabins.

---

### 3. Scatter Plot: Passenger Age vs. Ticket Fare Paid
- **File:** `charts/03_age_vs_fare_scatter.png`
- **Plot Type:** Multi-variable Scatter Plot (X: Age, Y: Fare, Hue: Survival, Style: Class, Outlier Annotations).
- **Empirical Findings:**
  - The vast majority of passengers clustered heavily below the £50 fare mark across all age brackets from 15 to 45 years.
  - **Extreme Fare Outlier Annotation:** Three passengers paid the all-time maximum ticket fare of **£512.33** (Miss Anna Ward, Mr. Thomas Drake Martinez Cardeza, and Mr. Gustave J. Lesurer). All three were 1st class passengers and all three survived.
  - High-fare ticket holders (£100+) show an overwhelming proportion of green points (survivors).

---

### 4. Histogram: Passenger Age Distribution with KDE
- **File:** `charts/04_age_distribution_histogram.png`
- **Plot Type:** Histogram (30 bins) with overlaid Kernel Density Estimation (KDE) curve.
- **Empirical Findings:**
  - **Total Non-Null Age Records:** 714 (177 missing entries, 19.87%).
  - **Mean Age:** **29.70 years** (red dashed line).
  - **Median Age:** **28.00 years** (green dash-dotted line).
  - **Standard Deviation:** **14.53 years**.
  - **Age Range:** Youngest was an infant of **0.42 years (~5 months old)**; oldest was **80.00 years old** (Algernon Barkworth, who survived).
- **Analytical Takeaway:**
  The distribution is unimodal with positive skewness, peaking sharply between 20 and 30 years old (young workers, laborers, and emigrants). A distinct secondary bump occurs between 0 and 5 years, representing infants and young children traveling with families.

---

### 5. Histogram: Fare Distribution Skewness Analysis
- **File:** `charts/05_fare_distribution_histogram.png`
- **Plot Type:** Side-by-side Dual Subplot (Panel A: Raw Fare; Panel B: Log-Transformed Fare).
- **Empirical Findings:**
  - **Raw Fare:** Extreme right skewness (positive skew). Median fare was just **£14.45**, with 75% of passengers paying **≤ £31.00**, yet the upper tail extends all the way to **£512.33**.
  - **Log-Transformed Fare:** Applying $\ln(1 + \text{Fare})$ compresses the extreme tail and stabilizes variance, transforming the distribution into a much more symmetrical, bell-shaped distribution suitable for parametric modeling.

---

### 6. Box Plot: Ticket Fare by Class (Outlier Detection)
- **File:** `charts/06_fare_by_class_boxplot.png`
- **Plot Type:** Categorical Box Plot with Tukey's 1.5 × IQR fences and custom outlier markers (red diamonds).
- **Empirical Findings:**
  - **Statistical Outlier Boundary (Tukey's Rule):**
    $$\text{Upper Fence} = Q_3 + 1.5 \times (Q_3 - Q_1) = 31.00 + 1.5 \times 23.09 = £65.63$$
  - **Total Fare Outliers Identified:** **116 passengers (13.02% of the dataset)** paid fares exceeding £65.63.
  - **Class Dispersions:**
    - **1st Class:** Median £60.29, IQR spans from £34.24 to £93.50, with massive dispersion and dozens of high-value outliers up to £512.33.
    - **2nd Class:** Median £14.25, tightly bounded between £13.00 and £26.00.
    - **3rd Class:** Median £8.05, extremely compressed with nearly all passengers paying between £7.75 and £15.50.

---

### 7. Box Plot: Passenger Age by Class and Survival Status
- **File:** `charts/07_age_by_class_and_survival_boxplot.png`
- **Plot Type:** Grouped Box Plot comparing age distributions across classes by survival outcome.
- **Empirical Findings:**
  - **1st Class Passengers:** Noticeably older (median age ~38.0 years), consistent with established financial wealth.
  - **2nd Class Passengers:** Median age ~29.0 years.
  - **3rd Class Passengers:** Youngest cohort (median age ~24.0 years).
  - **Child Priority Effect:** In 2nd and 3rd classes, the lower whiskers and lower quartile of survivors extend significantly lower toward infant ages compared to non-survivors, demonstrating that young children in lower decks were prioritized for lifeboats over adult peers.

---

### 8. Correlation Heatmap: Numerical Feature Correlations
- **File:** `charts/08_correlation_heatmap.png`
- **Plot Type:** Annotated Square Heatmap with divergent color map (`coolwarm`), range $[-1.0, +1.0]$.
- **Correlation Matrix Values:**
  - **`pclass` vs. `survived`:** **$-0.34$** (Strongest negative correlation — lower class number / higher status strongly correlates with survival).
  - **`fare` vs. `survived`:** **$+0.26$** (Strongest positive correlation — higher ticket fare correlates with higher survival probability).
  - **`pclass` vs. `fare`:** **$-0.55$** (Strong negative correlation — 1st class tickets cost significantly more).
  - **`sibsp` vs. `parch`:** **$+0.41$** (Moderate positive correlation — passengers traveling with siblings/spouses also tended to travel with parents/children).
  - **`pclass` vs. `age`:** **$-0.37$** (Older passengers were concentrated in higher ticket classes).

---

### 9. Executive Dashboard: Multi-Panel Summary
- **File:** `charts/09_data_visualization_dashboard.png`
- **Plot Type:** 2 × 2 Multi-subplot Executive Summary Dashboard.
- **Components:**
  1. Top-Left: Survival Rate by Class & Gender bar chart.
  2. Top-Right: Passenger Age Distribution histogram with KDE and median marker.
  3. Bottom-Left: Fare Outliers box plot across ticket classes.
  4. Bottom-Right: Age vs. Fare scatter plot colored by survival outcome.
- **Value:** Delivers an integrated, presentation-ready overview suitable for academic submissions, slide decks, or executive briefings.

---

## 🎨 Visualization Techniques & Best Practices

1. **Consistent Aesthetic Theme:** Configured global styles with `sns.set_theme(style="whitegrid")` for clean, professional contrast.
2. **Accessible Color Palettes:**
   - Gender: Coral (`#e76f51`) for Female and Deep Teal (`#2a9d8f`) for Male.
   - Survival: Vibrant Green (`#2ecc71`) for Survived and Crimson (`#e74c3c`) for Did Not Survive.
3. **Information Density & Annotations:** Included direct percentage numbers on bars and a targeted callout arrow pointing out the £512 luxury suite outlier.
4. **Statistical Rigor:** Benchmarked visual metrics against mathematical standards (Tukey's IQR outlier fences, KDE probability density curves, Pearson correlation coefficients).
5. **Robust Cross-Platform Output:** Formatted with standard UTF-8 console output encoding to eliminate Windows codepage crashes.
6. **Publication-Quality Exports:** Generated at **300 DPI** using `bbox_inches="tight"` to prevent label clipping.

---

## 💡 Conclusion & Key Findings
1. **The "Women and Children First" Protocol Was Strictly Enforced:** Gender was the dominant predictor of survival. Female survival was **74.20%**, compared to **18.89%** for males.
2. **Socio-Economic Privilege Directly Influenced Survival:** Passengers in 1st Class enjoyed a **62.96%** survival rate, compared to **47.28%** in 2nd Class and just **24.24%** in 3rd Class. 1st class women experienced near-total survival (**96.81%**).
3. **Severe Wealth Skewness:** Ticket fares exhibited massive right-skewness. Over 75% of passengers paid £31 or less, but the top 13% paid outlier fares reaching as high as £512.33.
4. **Port Demographics Correlated with Class:** Cherbourg passengers experienced higher survival (55.36%) because a larger proportion boarded into 1st class cabins.

---

## 👨‍🎓 Assignment Metadata
- **Assignment Name:** Data Visualization with Matplotlib and Seaborn
- **Project Name:** `Data_Visualization_Project`
- **Status:** Completed, verified, and ready for review.
