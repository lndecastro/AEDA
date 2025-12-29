# 2 — Data Description and Preparation

Before any meaningful exploratory analysis, visualization, or AI-supported insight generation can occur, data must be **understood, described, and prepared**. This module introduces the conceptual and practical foundations required to transform raw data into analyzable data.

This module builds directly on Module 1 by moving from the analytical process flow to the first operational stage of AEDA: understanding what the data is, how it is structured, and how it should be prepared for analysis.

## Learning Objectives

By the end of this module, students will be able to:

1. Represent datasets using tabular and mathematical forms.
2. Interpret and construct data dictionaries.
3. Classify data according to structure, nature, type, time variability, dimensionality, and ownership.
4. Recognize benchmark datasets commonly used in exploratory data analysis.
5. Identify common data quality problems in raw datasets.
6. Apply fundamental data preparation techniques such as sampling, missing-value handling, and normalization.

## 2.1 Tabular and Mathematical Representations

### Data, Information, Knowledge

**Data** can be understood as any element that can be stored, transferred, processed, or transformed to convey meaning, including numbers, words, images, sounds, and graphs. From a computational perspective, anything that can be digitally stored or manipulated qualifies as data. On its own, data has no inherent meaning; it becomes **information** only when placed within a context. When this information is interpreted, analyzed, and applied to support decisions or generate insights, it evolves into **knowledge**.

![Data, Information, Knowledge](./Data/DIKnowledge.png) <p>
**Figure:** From data to information to knowledge.

### Tabular Representation

The **tabular representation** is the most common way to organize data in exploratory data analysis.

- Rows represent objects, instances, patterns, records, or observations.
- Columns represent variables, attributes, or features.
- Each cell contains the value of a variable for a given object.

Tabular data structures are the foundation of spreadsheets, relational databases, BI tools, and data science libraries.

**Table:** First four objects of the Car Evaluation Dataset available at UCI.<p>
| Car ID | Buying | Maintenance | Doors | Persons | Lug_boot | Safety | Class |
|--------|--------|-------------|-------|---------|----------|--------|-------|
| 1      | vhigh  | vhigh       | 2     | 2       | small    | low    | unacc |
| 2      | vhigh  | vhigh       | 2     | 2       | small    | med    | unacc |
| 3      | vhigh  | vhigh       | 2     | 2       | small    | high   | unacc |
| 4      | vhigh  | vhigh       | 2     | 2       | med      | low    | unacc |


### Mathematical Representation

Mathematically, a dataset can be represented as a matrix:

$S = {x_i \mid i = 1, \ldots, N},$

where each object \(x_i\) is an \(m\)-dimensional vector given by

$\mathbf{x^i} = \big[ x_j^i \big]_{j = 1, \ldots, m}.\$

where:
- $N$ is the number of objects
- $m$ is the number of variables

This representation supports statistical analysis, linear algebra operations, optimization, and machine learning algorithms.

![Graph for Variables](./Data/Figure_2_1_Graph_for_variables.jpg) <p>
**Figure:** Graph for variables ‘safety’ vs ‘lug_boot’ of the four objects in the previous table.

## 2.2 Data Dictionary

A **data dictionary** provides a structured description of a dataset’s variables and their semantics. Typical elements include:

- Variable name
- Description and meaning
- Data type
- Valid values or ranges
- Units of measurement
- Missing-value indicators

### Why Data Dictionaries Matter

Data dictionaries:
- Reduce ambiguity and misinterpretation
- Improve reproducibility
- Support collaboration across teams
- Provide essential context for automated analysis and AI-assisted workflows

**Table:** Example of a simple data dictionary for the Car Evaluation Dataset
| Variable name | Definition (meaning)                  | Domain                         |
|---------------|---------------------------------------|--------------------------------|
| Car ID        | ID number of each car in the dataset  | Integer number                 |
| Buying        | Buying price                          | {v-high, high, med, low}       |
| Maintenance   | Price of the maintenance              | {v-high, high, med, low}       |
| Doors         | Number of doors                       | {2, 3, 4, 5-more}              |
| Persons       | Number of persons accommodated        | {2, 4, more}                   |
| Lug-boot      | Trunk size                            | {small, med, big}              |
| Safety        | Level of safety                       | {low, med, high}               |
| Class         | Car acceptability                     | {unacc, acc, good, vgood}      |

## 2.3 Classifying Data

Data can be classified along multiple dimensions. These classifications guide analytical decisions, visualization choices, and preparation strategies.

### 2.3.1 Structure

- **Structured data**: Fixed schema (tables, spreadsheets, relational databases)
- **Semi-structured data**: Flexible schema with metadata (JSON, XML)
- **Unstructured data**: No predefined structure (text, images, audio, video)

### 2.3.2 Nature

- **Quantitative data**: Numerical measurements
- **Qualitative data**: Descriptive or categorical attributes

### 2.3.3 Type

- **Numerical data**: Discrete or continuous values
- **Categorical data**: Nominal or ordinal categories

### 2.3.4 Time Variability

- **Invariant (static) data**: Values do not change over time
- **Time-varying data**: Values evolve across time

### 2.3.5 Dimensionality

- **Unidimensional data**: One variable of interest
- **Multidimensional data**: Multiple variables analyzed jointly

### 2.3.6 Ownership

- **Public data**: Freely available datasets
- **Private data**: Restricted or proprietary datasets

---

## 2.4 Datasets Used in the Course

The following datasets are used throughout the course to illustrate AEDA concepts:

- Mammographic Dataset
- Forest Fires Dataset
- Iris Dataset of Fisher
- Auto MPG Dataset
- Gapminder Dataset
- NaturalEarth Low Resolution Dataset
- Daily Delhi Climate Train Dataset
- IMDb Movie Reviews Dataset
- Zachary’s Karate Club Dataset

These datasets span different domains, data types, and analytical challenges.

---

## 2.5 Data Preparation

### Introduction

**Raw data** is data in its original form, collected directly from sources such as sensors, surveys, databases, or logs. Raw data is rarely suitable for immediate analysis.

Common problems include:
- Data overload
- Missing values
- Inconsistencies
- Noise

Data preparation addresses these issues and ensures data quality and usability.

---

### 2.5.1 Sampling

Sampling reduces data volume while preserving representativeness.

Common sampling strategies include:
- Random sampling with replacement
- Random sampling without replacement
- Systematic sampling
- Group sampling
- Stratified sampling

Sampling is essential for scalability and efficient exploratory analysis.

---

### 2.5.2 Missing Values

Missing data may result from sensor failures, human error, or integration issues.

Common strategies include:
- Removing affected objects
- Manual imputation
- Global constant substitution
- Hot-deck imputation
- Central tendency imputation (mean, median, mode)
- Class-conditional imputation

The choice of strategy depends on data type, context, and analytical goals.

---

### 2.5.3 Normalization

Normalization transforms numerical variables to a common scale.

Common techniques include:
- Min–max normalization
- Z-score standardization
- Scaling by maximum value

Normalization improves interpretability, visualization, and the performance of distance-based and learning-based methods.

---

## Reflection

- How does data classification influence analytical and visualization choices?
- What risks arise from inadequate data preparation?
- How does data preparation support responsible and effective AI-assisted analysis?

---

## Further Reading

Refer to the course syllabus bibliography, particularly:

**De Castro, L. N. (2026). _Exploratory Data Analysis: Descriptive Analysis, Visualization, and Dashboard Design_. CRC Press.**

---

*End of Module 2*
