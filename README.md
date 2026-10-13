SWYNEX Task 1: Data Cleaning and Preparation

About the Project

For this task, I worked with a food waste prediction dataset from Kaggle. The dataset contains information about meals served, kitchen staff, temperature, humidity, staff experience, and food waste.

Before working with the data, I wanted to understand what needed fixing and how I could make it more consistent and reliable for further analysis.

Tools Used

Python, Pandas, NumPy, VS Code, Git and GitHub.

What I Worked On

- Checked the dataset for missing values and duplicate records.
- Found 164 missing values in the staff_experience column.
- Standardized inconsistent labels such as "MeAt", "MEAT", and "dairy".
- Replaced missing staff experience with "Unknown" instead of guessing the values.
- Converted the date and numeric columns to appropriate data types.
- Investigated unusually high meal counts and food waste measurements.
- Created reports to document the cleaning results and records that may need further review.

Results

- Records before cleaning: 911
- Records after cleaning: 911
- Columns retained: 12
- Missing values after cleaning: 0
- Exact duplicate rows after cleaning: 0
- Missing staff experience values labelled as Unknown: 164

Files Created

- clean_data.py: Python script used for cleaning the dataset.
- data/processed/food_waste_cleaned.csv: Final cleaned dataset.
- reports/cleaning_summary.csv: Summary of the before-and-after results.
- reports/missing_values_after.csv: Missing-value report.
- reports/records_for_review.csv: Records flagged for further investigation.
- reports/duplicate_ids_review.csv: Duplicate identifier review.
- reports/unexpected_categories.csv: Category validation report.
- reports/unexpected_experience_values.csv: Staff experience validation report.

How to Run the Project

1. Install Python 3.
2. Install the required libraries using: python -m pip install -r requirements.txt
3. Place the original dataset in data/raw/train.csv.
4. Run the script using: python clean_data.py
5. Check the cleaned dataset in data/processed and review the reports folder.

What I Learned

This task helped me understand that data cleaning involves more than removing duplicates or filling empty cells. It is important to investigate unusual values, keep valid records, and make decisions that do not introduce misleading information.

I also learned the importance of documenting the changes made to a dataset so that the results can be checked and reproduced.

Dataset Source

Kaggle – Messy Food Waste Prediction Dataset

https://www.kaggle.com/competitions/messy-food-waste-prediction-dataset
