# Food Waste Exploratory Data Analysis

## Project Overview

Food waste is a practical problem for kitchens and food service businesses. Understanding how much food gets wasted, which categories contribute the most, and how waste varies with daily operations can help identify areas that deserve closer attention.

For this project, I worked with a food waste prediction dataset containing kitchen activity, environmental conditions, staff experience, and food waste measurements. I used Python to explore the cleaned data, compare different groups, identify unusual records, and present the findings through charts.

## Dataset Details

- **Dataset:** Messy Food Waste Prediction Dataset
- **Records analysed:** 911
- **Columns:** 12
- **Date range:** January 2022 to September 2024
- **Target measure:** Food waste in kilograms

The dataset includes meals served, kitchen staff, temperature, humidity, day-of-week codes, special events, previous food waste, staff experience, and waste categories.

## Tools and Technologies

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- VS Code
- Git and GitHub

## What I Worked On

I explored the dataset using descriptive statistics, grouped comparisons, correlation analysis, and visualizations.

The analysis covered:

- Average food waste across Meat, Vegetables, Dairy, and Grains.
- The relationship between meals served and food waste.
- Food waste distributions across staff experience groups.
- Temperature and its association with food waste.
- Differences in average waste across day-of-week codes.
- Unusually high or low food waste values using the IQR method.

## Key Findings

**1. Waste by category**

Meat had the highest average food waste at 46.30 kg, followed by Vegetables at 45.85 kg. Grains had the lowest average at 42.00 kg.

**2. Meals served and food waste**

The correlation between meals served and food waste was 0.852, indicating a strong positive linear association in the dataset. This does not, by itself, establish causation.

**3. Staff experience**

Records labelled Beginner had the highest observed average food waste at 47.85 kg. This difference is worth investigating, but it does not prove that staff experience causes food waste.

**4. Day-of-week comparison**

Day code 2 had the highest average waste at 47.06 kg, while day code 1 had the lowest at 42.33 kg. The original code-to-weekday mapping was not assumed.

**5. Unusual food waste records**

The IQR method flagged 24 records for review. These records were retained for investigation rather than automatically treated as errors.

## Project Structure

- `data/` — cleaned dataset used for the analysis
- `charts/` — generated visualizations
- `reports/` — grouped statistics, insights, and outlier review
- `eda.py` — Python analysis script
- `requirements.txt` — Python dependencies

## How to Run the Project

Clone or download the repository, then open a terminal in this project folder.

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the analysis:

```bash
python eda.py
```

The script generates the charts and reports in the `charts/` and `reports/` folders.

## Conclusion

This project helped me practise working with a real-world-style dataset, summarizing data with Pandas, interpreting correlations, comparing categories, identifying potential outliers, and communicating findings through visualizations.

The findings provide a starting point for investigating food waste patterns. Further analysis would be needed to validate unusual records and determine which operational factors are genuinely associated with waste.

## Author

Harvansh Singh Dhanjal

GitHub: https://github.com/Harvansh26