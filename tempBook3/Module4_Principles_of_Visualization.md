# 4 — Principles of Data Visualization

This module introduces the **principles that govern how humans perceive, interpret, and make sense of visual information**, and how these principles translate into effective data visualizations. While previous modules focused on *what* to compute and summarize, this module focuses on *how to visually communicate data* in a way that supports cognition, reduces misinterpretation, and enables insight.

The concepts presented here provide the perceptual and design foundation for the visualization methods to be explored in subsequent modules.

## Learning Objectives

By the end of this module, students will be able to:

1. Explain how human visual perception affects data interpretation.
2. Distinguish between attentive and preattentive visual processing.
3. Apply Gestalt principles to analyze and design visualizations.
4. Evaluate tables and graphs based on fundamental visualization design principles.
5. Identify common visualization pitfalls and cognitive overload issues.

## 4.1 Visual Processing

Data visualization is fundamentally a **human–computer interaction problem**. Visual representations must align with how the human visual system processes information. Poorly designed visualizations increase cognitive load and obscure patterns, while well-designed ones exploit perceptual strengths.

### 4.1.1 “Can’t See the Forest for the Trees”

This expression captures a frequent problem in data visualization: **excessive detail prevents global understanding**.

Common causes include:

* Too many variables displayed simultaneously
* Excessive annotations, labels, or colors
* Overly granular data without aggregation

The human brain has limited working memory. When visual complexity exceeds this limit, users focus on local details and fail to perceive global structure, trends, or anomalies.

**Design implication:**
Visualization should emphasize *structure before detail*. Aggregation, filtering, and layering are essential strategies to guide attention.

Another important aspect in EDA discussed here is that while summary measures are very useful for the understanding of data, it is usually insufficient for us to have full knowledge of the data distribution. More specifically, it is possible that some data sets present the same summary measures but significantly different structure and distributions. 

To illustrate, consider the case of the Anscombe's Quartet presented in the code snippet and figure below. Despite some structural differences, all datasets have the same summary measures and regression lines.

```python
# CODE 4.1
# Print the Anscombe's Quartet table with the summary measures (mean, std, corr, linear 
# regression) for each dataset and plot the scatterplots of each of the four datasets

import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Uses the Seaborn library to load the Anscombe's data 
danscombe = sns.load_dataset("anscombe")
pd.options.display.float_format = "{:.2f}".format

# Print the data and summary measures for each dataset
for dataset in ["I", "II", "III", "IV"]:
    df_subset = danscombe[danscombe.dataset == dataset]
    print(f"Dataset  {dataset}\n{df_subset}")
    print(f"Summary Measures for Dataset {dataset}:")
    print(f"Mean of x: {np.mean(df_subset.x):.2f}")
    print(f"Mean of y: {np.mean(df_subset.y):.2f}")
    print(f"Std of x: {np.std(df_subset.x):.2f}")
    print(f"Std of y: {np.std(df_subset.y):.2f}")
    print(f"Correlation between x and y: {np.corrcoef(df_subset.x, df_subset.y)[0,1]:.2f}")
    model = sm.OLS(df_subset.y, sm.add_constant(df_subset.x)).fit()
    print(f"Linear regression model: y = {model.params.iloc[0]:.2f} + {model.params.iloc[1]:.2f}x\n")

# Plot the scatterplots and regression lines for each dataset
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for i, dataset in enumerate(["I", "II", "III", "IV"]):
    ax = axes.flatten()[i]
    df_subset = danscombe[danscombe.dataset == dataset]
    x = df_subset.x; y = df_subset.y
    model = sm.OLS(y, sm.add_constant(x)).fit()
    y_pred = model.predict(sm.add_constant(x))
    sns.scatterplot(x=x, y=y, ax=ax)
    sns.lineplot(x=x, y=y_pred, color="red", ax=ax)
    ax.set_title(f"Dataset {dataset}", fontsize=16)

# Save & Show the plot
plt.savefig("Figure_4_1_Anscombe_Quartet.svg", format="svg", dpi=1500)
plt.show()
```

![Anscombe's Quartet](./Data/Figure_4_1_Anscombe_Quartet.jpg)

### Prompt — Anscombe's Quartet Table with Summary Measures
```
You are a data analysis assistant supporting an **Advanced Exploratory Data Analysis (AEDA)** course.

Your goal is to **perform the descriptive and visual analysis of Anscombe’s Quartet**. This task highlights how datasets with identical summary statistics can exhibit radically different visual patterns.

## Dataset

- Load **Anscombe’s Quartet** using the Seaborn built-in dataset loader.
- Treat the data exactly as provided (no preprocessing or transformations).
- Ensure all numerical outputs are displayed with **two decimal places**.

## Tasks

### Part 1 — Descriptive Summary for Each Dataset

For each dataset, perform the following steps independently:

1. Subset the data to include only the current dataset.
2. Display the full table of `(x, y)` values for that dataset.
3. Compute and report the following **summary measures**:
   - Mean of `x`
   - Mean of `y`
   - Standard deviation of `x`
   - Standard deviation of `y`
   - Pearson correlation coefficient between `x` and `y`
4. Fit a **simple linear regression model** using **ordinary least squares (OLS)**.
5. Report the regression equation using the estimated coefficients.
6. Present all results in a clear, labeled, and readable textual format.

### Part 2 — Visualization

1. Create a single figure composed of a **2 × 2 grid of subplots**.
2. For each dataset (I–IV):
   - Plot a **scatterplot** of `x` versus `y`.
   - Overlay the **corresponding linear regression line**.
   - Assign the subplot title as `Dataset I`, `Dataset II`, etc.
3. Use consistent visual styling across all subplots to allow direct comparison.

## Output Requirements

- Printed output must include:
  - The full table for each dataset
  - All computed summary measures
  - The linear regression equation for each dataset
- The final figure must:
  - Contain four subplots (one per dataset)
  - Clearly show both data points and regression lines
```

An example similar to the **Anscombe’s Quartet** is the **Datasaurus Dozen**, a dataset consisting of 13 distinct datasets, each with a different shape, but with practically the same summary measures. The code snippet below is similar to the previous one, but reads the Datasaurus Dozen dataset from a CSV file with semicolon delimiter using the Pandas library. 

```python
# CODE 4.2
# Print the Datasaurus Dozen table with the summary measures (mean, std, corr, linear 
# regression) and plot the scatterplots of each dataset

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset using Pandas
dsaurus = pd.read_csv("datasaurus_dozen.csv", delimiter=';')
pd.options.display.float_format = "{:.2f}".format

# Print the data and summary measures for each dataset
for dataset in dsaurus.dataset.unique():
    print(f"Summary Measures for Dataset {dataset}:")
    df_subset = dsaurus[dsaurus.dataset == dataset]
    print(f"Mean of x: {np.mean(df_subset.x):.2f}")
    print(f"Mean of y: {np.mean(df_subset.y):.2f}")
    print(f"Std of x: {np.std(df_subset.x):.2f}")
    print(f"Std of y: {np.std(df_subset.y):.2f}")
    print(f"Correlation between x and y: {np.corrcoef(df_subset.x, df_subset.y)[0, 1]:.2f}")
    model = sm.OLS(df_subset.y, sm.add_constant(df_subset.x)).fit()
    print(f"Linear regression model: y = {model.params.iloc[0]:.2f} + {model.params.iloc[1]:.2f}x\n")

# Plot the scatterplots for each dataset
fig, axes = plt.subplots(5, 3, figsize=(14, 18))

for i, dataset in enumerate(dsaurus.dataset.unique()):
    ax = axes.flatten()[i]
    df_subset = dsaurus[dsaurus.dataset == dataset]
    x = df_subset.x; y = df_subset.y
    sns.scatterplot(x=x, y=y, ax=ax)
    ax.set_title(f"Dataset {dataset}", fontsize=16)
    
for i in range(len(dsaurus.dataset.unique()), len(axes.flatten())):
    fig.delaxes(axes.flatten()[i])  # Remove unused subplots

# Save & Show the plot
plt.savefig("Figure_4_2_Datasaurus_Dozen.svg", format="svg", dpi=1500)
plt.show()
```

Summary Measures for Dataset dino:  <br>
Mean of x: 54.26 <br>
Mean of y: 47.83  <br>
Std of x: 16.71  <br>
Std of y: 26.84  <br>
Correlation between x and y: -0.06 <br>
Linear regression model: y = 53.45 + -0.10x. <p>

![Datasauros Dozen](./Data/Figure_4_2_Datasaurus_Dozen.jpg) <p>
**Figure:** Scatterplots of the 13 datasets available in the Datasaurus Dozen.

### 4.1.2 Preattentive Processing

**Preattentive processing** refers to the ability of the human visual system to rapidly and automatically detect certain visual features *before conscious attention is engaged*.

Preattentive attributes include:

* Color (hue, intensity)
* Orientation
* Size
* Shape
* Position
* Motion

These attributes are processed in parallel and enable near-instant detection of patterns such as outliers or clusters.

For example:

* A red dot among gray dots is immediately visible
* A larger bar among equal-sized bars stands out

**Design implication:**
Key variables should be mapped to preattentive attributes when rapid detection is desired.

---

### 4.1.3 Gestalt Principles and Data Visualization Methods

Gestalt psychology explains how humans naturally group visual elements. These principles are critical in visualization design.

Key Gestalt principles include:

* **Proximity**: Elements close to each other are perceived as related.
* **Similarity**: Elements sharing color, shape, or size are grouped together.
* **Continuity**: The eye follows smooth paths and trends.
* **Closure**: Incomplete shapes are perceived as complete.
* **Figure–ground**: Objects are distinguished from their background.

**Design implication:**
Effective visualizations intentionally leverage these principles to encode relationships without explicit annotation.

---

## 4.2 Design Principles for Data Visualization

Beyond perception, visualization requires **design discipline**. Design principles ensure clarity, accuracy, and interpretability.

Core principles include:

* Simplicity
* Consistency
* Accuracy
* Integrity of scales
* Appropriate encoding

---

### 4.2.1 Tables

Tables are precise but cognitively demanding. They are best suited for:

* Exact values
* Lookup tasks
* Small to medium datasets

**Design guidelines for tables:**

* Align numbers by decimal place
* Use consistent units
* Avoid unnecessary gridlines
* Use whitespace to separate logical groups
* Order rows and columns meaningfully

Tables should support comparison *only when visual alternatives are insufficient*.

---

### 4.2.2 Graphs

Graphs leverage visual perception to reveal patterns, trends, and relationships.

Effective graph design requires:

* Choosing the correct chart type for the data
* Proper scaling (avoiding misleading axes)
* Clear labeling and legends
* Minimal chart junk

Common graph categories include:

* Distribution (histograms, boxplots)
* Association (scatterplots, heatmaps)
* Comparison (bar charts)
* Composition (pie charts, treemaps)
* Temporal evolution (line charts)

**Design implication:**
The same dataset can produce radically different interpretations depending on the chosen visualization.

---

## Reflection

* How does visual perception constrain what can be effectively communicated?
* Which preattentive attributes are most effective for highlighting anomalies?
* When should tables be preferred over graphs?
* How can Gestalt principles reduce the need for explicit explanation?

---

## Further Reading

Students are encouraged to consult the bibliography listed in the course syllabus, particularly the references on data visualization, storytelling, and perceptual foundations.
