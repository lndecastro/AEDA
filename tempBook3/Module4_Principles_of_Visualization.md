# Module 4 — Principles of Data Visualization

This module introduces the **principles that govern how humans perceive, interpret, and make sense of visual information**, and how these principles translate into effective data visualizations. While previous modules focused on *what* to compute and summarize, this module focuses on *how to visually communicate data* in a way that supports cognition, reduces misinterpretation, and enables insight.

The concepts presented here provide the perceptual and design foundation for the visualization methods explored in subsequent modules.

---

## Learning Objectives

By the end of this module, students will be able to:

1. Explain how human visual perception affects data interpretation.
2. Distinguish between attentive and preattentive visual processing.
3. Apply Gestalt principles to analyze and design visualizations.
4. Evaluate tables and graphs based on fundamental visualization design principles.
5. Identify common visualization pitfalls and cognitive overload issues.

---

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

---

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
