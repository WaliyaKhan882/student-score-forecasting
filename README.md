# Student Score Forecasting

How early in a semester can we predict a student's final score? This project forecasts each student's final course score (out of 100) from only the assessments completed so far, and measures how the forecast improves as more quizzes, assignments and exams come in.

The goal is early decision support: flag students who may need help while there is still time to act. Teachers keep the final judgment.

---

## Approach

- **Data:** anonymised result sheets for two courses, Cloud Computing and ICT, each with a morning and an afternoon section. Student IDs and names are placeholders (S1, N1, ...).
- **Train and test split:** the model learns from the morning section and is tested on the afternoon section, so it is always predicting students it has never seen.
- **Incremental forecasting:** starting after the 5th assessment, a linear regression model is retrained on the first *k* assessments and predicts the final score. This repeats for every *k* up to the second-last assessment.
- **Evaluation:** for every student, the average, maximum and minimum absolute error across all forecast points.
- **Stabilisation point:** the step where the forecast error drops the most in a single step, which shows when the forecast becomes reliable.

## Results

| Course | Average absolute error (points out of 100) | Forecast becomes reliable after |
|---|---|---|
| Cloud Computing | 5.7 | the 8th assessment, where mean error falls from 8.2 to 3.9 |
| ICT | 4.7 | the 6th assessment, where mean error falls from 9.2 to 4.4 |

In both courses the big jump comes when the midterm enters the model, since it carries a large share of the final grade. Full tables are in `results/`.

---

## Run it

```bash
pip install -r requirements.txt
python forecast_final_scores.py
```

The script prints a summary and writes three files to `results/`:

| File | Contents |
|---|---|
| `predictions.xlsx` | Actual score and the forecast after each assessment, per student |
| `error_tables.xlsx` | Per-student average, maximum and minimum absolute error |
| `analysis.xlsx` | Error trend by assessment and student difficulty groups |

## Repository contents

```
student-score-forecasting/
├── forecast_final_scores.py
├── requirements.txt
├── data/
│   ├── cloud_computing_results.xlsx
│   └── ict_results.xlsx
└── results/
```

## Limits

Two courses and a few dozen students per section is a small sample, and a linear model cannot capture every pattern. Results like these should support a conversation with a student, not replace one.

---

Built by [Waliya Khan](https://github.com/WaliyaKhan882) during my MS in Data Science at the University of the Punjab.
