"""
Final Score Forecasting from Partial Assessment Activities
=========================================================================
Training set : Morning session (each course)
Testing set  : Afternoon session (each course)
Prediction   : Final score starting after every 5th activity onward

Courses:
  1. Cloud Computing (CC)   — data/cloud_computing_results.xlsx
  2. ICT                    — data/ict_results.xlsx
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings, os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_sheet(path, sheet_name):
    """
    Excel layout (0-indexed rows):
      Row 0  – Weights/Scale
      Row 1  – Column names (Sr. No., Roll Number, Name, A1, Q1, …, Final, Total/NaN)
      Row 2  – Max marks
      Row 3+ – Student records

    Returns cleaned DataFrame with proper column names.
    The LAST column is always the scaled total out of 100 — we rename it 'Total'.
    The first 3 columns are always: Sr, RollNumber, Name (even if NaN in header).
    """
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)

    # Row 1 → column names
    col_names = raw.iloc[1].tolist()

    # Force-name the first 3 metadata columns and the last (Total) column
    col_names[0] = "Sr"
    col_names[1] = "RollNumber"
    col_names[2] = "Name"
    col_names[-1] = "Total"

    df = raw.iloc[3:].copy()
    df.columns = col_names
    df = df.reset_index(drop=True)

    # Drop rows where RollNumber is missing
    df = df.dropna(subset=["RollNumber"])

    # Convert Total to numeric
    df["Total"] = pd.to_numeric(df["Total"], errors="coerce")

    return df, "RollNumber"


def get_activity_cols(df, roll_col):
    """
    Activity columns = everything except Sr, RollNumber, Name, Total.
    """
    skip = {"sr", "rollnumber", "name", "total"}

    activity_cols = []
    for c in df.columns:
        cs = str(c).strip().lower().replace(" ", "").replace(".", "")
        if cs in skip:
            continue
        if isinstance(c, float) and np.isnan(c):
            continue
        activity_cols.append(c)

    return activity_cols


# ─────────────────────────────────────────────────────────────
# 2.  PREDICTION
# ─────────────────────────────────────────────────────────────

def predict_incrementally(train_df, test_df, activity_cols, id_col, start_from=5):
    """
    For k = start_from to n-1 (second-last activity):
      - Train LinearRegression on first k activity columns.
      - Predict 'Total' for each test student.

    Returns:
      results_df       : StudentID | ActualScore | Pred_after_5th | ...
      errors_df        : StudentID | AvgAE | MaxAE | MinAE  (+ AVERAGE row)
      per_activity_df  : activity index | mean absolute error across all students
    """
    n = len(activity_cols)
    prediction_labels = []
    all_preds = {row[id_col]: [] for _, row in test_df.iterrows()}

    for k in range(start_from, n):   # stop at n-1 = second-last activity
        features = activity_cols[:k]
        label = f"Pred_after_{k}th"
        prediction_labels.append(label)

        X_tr = train_df[features].apply(pd.to_numeric, errors="coerce").fillna(0).values
        y_tr = train_df["Total"].fillna(0).values
        X_te = test_df[features].apply(pd.to_numeric, errors="coerce").fillna(0).values

        model = LinearRegression()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        for i, (_, row) in enumerate(test_df.iterrows()):
            all_preds[row[id_col]].append(round(float(y_pred[i]), 2))

    # ── Results DataFrame ───────────────────────────────────
    results_rows = []
    for _, row in test_df.iterrows():
        sid = row[id_col]
        r = {id_col: sid,
             "ActualScore": round(float(row["Total"]), 2) if not pd.isna(row["Total"]) else 0}
        for j, lbl in enumerate(prediction_labels):
            r[lbl] = all_preds[sid][j]
        results_rows.append(r)
    results_df = pd.DataFrame(results_rows)

    # ── Per-student error DataFrame ─────────────────────────
    # Store raw per-round absolute errors for each student (used in analysis)
    student_ae_matrix = []   # rows = students, cols = rounds
    error_rows = []
    for _, row in results_df.iterrows():
        actual = row["ActualScore"]
        preds  = [row[lbl] for lbl in prediction_labels]
        ae     = [abs(actual - p) for p in preds]
        student_ae_matrix.append(ae)
        error_rows.append({
            id_col             : row[id_col],
            "AvgAbsoluteError" : round(float(np.mean(ae)), 4),
            "MaxAbsoluteError" : round(float(np.max(ae)),  4),
            "MinAbsoluteError" : round(float(np.min(ae)),  4),
        })

    errors_df = pd.DataFrame(error_rows)

    # Averages row
    avg_row = {
        id_col             : "AVERAGE",
        "AvgAbsoluteError" : round(errors_df["AvgAbsoluteError"].mean(), 4),
        "MaxAbsoluteError" : round(errors_df["MaxAbsoluteError"].mean(), 4),
        "MinAbsoluteError" : round(errors_df["MinAbsoluteError"].mean(), 4),
    }
    errors_df = pd.concat([errors_df, pd.DataFrame([avg_row])], ignore_index=True)

    # ── Per-activity mean error (across all students) ───────
    ae_matrix = np.array(student_ae_matrix)   # shape: (n_students, n_rounds)
    round_labels = [f"After_{start_from + i}th_activity" for i in range(ae_matrix.shape[1])]
    per_activity_df = pd.DataFrame({
        "Activity_Round" : round_labels,
        "MeanAbsError"   : np.round(ae_matrix.mean(axis=0), 4),
    })

    return results_df, errors_df, per_activity_df


# ─────────────────────────────────────────────────────────────
# ANALYSIS HELPERS
# ─────────────────────────────────────────────────────────────

def find_stabilization_point(per_activity_df):
    """
    Find the stabilization point using the 'biggest single drop' (elbow) method.

    The stabilization point is defined as the round AFTER which the largest
    single improvement in mean absolute error occurred. This is the round
    where the model gained the most predictive power in one step.

    Example for CC:
      5th→6th drop: 9.11 - 8.74 = 0.37
      6th→7th drop: 8.74 - 8.16 = 0.58
      7th→8th drop: 8.16 - 3.85 = 4.31  ← BIGGEST DROP
      → Stabilization point = After_8th_activity

    This avoids the flaw of the proximity-to-minimum approach which simply
    returns the global minimum round itself rather than the true elbow.
    """
    errors = per_activity_df["MeanAbsError"].values
    drops  = [errors[i - 1] - errors[i] for i in range(1, len(errors))]

    biggest_drop_idx = int(np.argmax(drops))   # index in drops array
    stab_idx = biggest_drop_idx + 1            # round AFTER the biggest drop
    biggest_drop_val = round(drops[biggest_drop_idx], 4)

    return stab_idx, per_activity_df["Activity_Round"].iloc[stab_idx], biggest_drop_val





def classify_students(errors_df, id_col):
    """
    Classify students as Easy / Moderate / Hard to predict
    using data-driven thresholds based on the distribution of AvgAbsoluteError.

    Thresholds are computed from the data itself:
      Easy threshold  = mean - 0.5 * std  (well below average error)
      Hard threshold  = mean + 0.5 * std  (well above average error)
      Moderate        = everything in between

    This avoids arbitrary hardcoded values and adapts to each course automatically.
    """
    df = errors_df[errors_df[id_col] != "AVERAGE"].copy()
    df["AvgAbsoluteError"] = pd.to_numeric(df["AvgAbsoluteError"], errors="coerce")

    mean_ae = df["AvgAbsoluteError"].mean()
    std_ae  = df["AvgAbsoluteError"].std()

    easy_threshold = mean_ae - 0.5 * std_ae
    hard_threshold = mean_ae + 0.5 * std_ae

    easy     = df[df["AvgAbsoluteError"] <= easy_threshold][id_col].tolist()
    hard     = df[df["AvgAbsoluteError"] >= hard_threshold][id_col].tolist()
    moderate = df[(df["AvgAbsoluteError"] >  easy_threshold) &
                  (df["AvgAbsoluteError"] <  hard_threshold)][id_col].tolist()

    return easy, moderate, hard, round(easy_threshold, 2), round(hard_threshold, 2)




# ─────────────────────────────────────────────────────────────
# 3.  MAIN
# ─────────────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.abspath(__file__))

COURSES = {
    "CloudComputing": {
        "file"        : os.path.join(BASE, "data", "cloud_computing_results.xlsx"),
        "train_sheet" : "cloud computing morning",
        "test_sheet"  : "cloud computing afternoon",
    },
    "ICT": {
        "file"        : os.path.join(BASE, "data", "ict_results.xlsx"),
        "train_sheet" : "ICT Morning",
        "test_sheet"  : "ICT Afternoon",
    },
}

all_results        = {}
all_errors         = {}
all_per_activity   = {}   # per-activity mean error trend
all_classification = {}   # student easy/moderate/hard classification

for course_name, info in COURSES.items():
    print(f"\n{'='*60}")
    print(f"  COURSE: {course_name}")
    print(f"{'='*60}")

    train_df, id_col = load_sheet(info["file"], info["train_sheet"])
    test_df,  _      = load_sheet(info["file"], info["test_sheet"])

    activity_cols = get_activity_cols(train_df, id_col)

    print(f"  Student ID col  : {id_col}")
    print(f"  Total activities: {len(activity_cols)}")
    print(f"  Activities      : {activity_cols}")
    print(f"  Train students  : {len(train_df)}  |  Test students: {len(test_df)}")

    results_df, errors_df, per_activity_df = predict_incrementally(
        train_df, test_df,
        activity_cols=activity_cols,
        id_col=id_col,
        start_from=5
    )

    all_results[course_name]        = results_df
    all_errors[course_name]          = errors_df
    all_per_activity[course_name]    = per_activity_df

    # ── Standard error table ──────────────────────────────
    print(f"\n  Error Table (StudentID | AvgAE | MaxAE | MinAE):")
    print(errors_df.to_string(index=False))

    # ── Per-activity error trend ──────────────────────────
    print(f"\n  Mean Absolute Error after each activity (all students averaged):")
    print(per_activity_df.to_string(index=False))

    stab_idx, stab_round, biggest_drop = find_stabilization_point(per_activity_df)
    print(f"\n  >> Stabilization point: {stab_round}")
    print(f"     (Biggest single drop in error = {biggest_drop} marks — largest gain in one step)")


    # ── Hard vs Easy student classification ──────────────
    easy, moderate, hard, easy_t, hard_t = classify_students(errors_df, id_col)
    print(f"\n  Student Prediction Difficulty:")
    print(f"    Thresholds (data-driven): Easy <= {easy_t}  |  Hard >= {hard_t}")
    print(f"    Easy     : {easy if easy else 'None'}")
    print(f"    Moderate : {moderate if moderate else 'None'}")
    print(f"    Hard     : {hard if hard else 'None'}")

    # Build classification DataFrame and store for saving
    classify_rows = (
        [{id_col: s, "Difficulty": "Easy"}     for s in easy]     +
        [{id_col: s, "Difficulty": "Moderate"} for s in moderate] +
        [{id_col: s, "Difficulty": "Hard"}     for s in hard]
    )
    classify_df = pd.DataFrame(classify_rows)
    all_classification[course_name] = classify_df



# ─────────────────────────────────────────────────────────────
# 4.  SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────

out = os.path.join(BASE, "results")
os.makedirs(out, exist_ok=True)

# Predictions Excel (2 sheets)
pred_path = os.path.join(out, "predictions.xlsx")
with pd.ExcelWriter(pred_path, engine="openpyxl") as w:
    for cname, df in all_results.items():
        df.to_excel(w, sheet_name=cname[:31], index=False)

# Error Tables Excel (2 sheets)
err_path = os.path.join(out, "error_tables.xlsx")
with pd.ExcelWriter(err_path, engine="openpyxl") as w:
    for cname, df in all_errors.items():
        df.to_excel(w, sheet_name=cname[:31], index=False)

# Analysis Excel — exactly 4 sheets:
#   CC_ActivityTrend  : per-activity mean error with stabilization point as last row
#   ICT_ActivityTrend : same for ICT
#   CC_Difficulty     : StudentID + Difficulty (Easy/Moderate/Hard)
#   ICT_Difficulty    : same for ICT
analysis_path = os.path.join(out, "analysis.xlsx")
with pd.ExcelWriter(analysis_path, engine="openpyxl") as w:

    # Activity Trend sheets — one per course, stabilization appended as last row
    for cname, act_df in all_per_activity.items():
        _, s_round, s_drop = find_stabilization_point(act_df)
        stab_row = pd.DataFrame([{
            "Activity_Round": "-- Stabilization Point --",
            "MeanAbsError"  : f"{s_round}  (biggest drop = {s_drop} marks at this step)"
        }])
        trend_sheet = pd.concat([act_df, stab_row], ignore_index=True)
        sname = "CC_ActivityTrend" if "Cloud" in cname else "ICT_ActivityTrend"
        trend_sheet.to_excel(w, sheet_name=sname, index=False)


    # Difficulty sheets — one per course, only StudentID + Difficulty columns
    for cname, cls_df in all_classification.items():
        id_col_name = cls_df.columns[0]
        clean_df = cls_df[[id_col_name, "Difficulty"]].copy()
        sname = "CC_Difficulty" if "Cloud" in cname else "ICT_Difficulty"
        clean_df.to_excel(w, sheet_name=sname, index=False)

print(f"\n[DONE] Predictions saved  -> {pred_path}")
print(f"[DONE] Error tables saved -> {err_path}")
print(f"[DONE] Analysis saved     -> {analysis_path}")



# Summary
print("\n\n" + "="*60)
print("  SUMMARY")
print("="*60)
for cname, df in all_errors.items():
    avg = df[df.iloc[:, 0] == "AVERAGE"].iloc[0]
    print(f"\nCourse: {cname}")
    print(f"  Avg of AvgAbsoluteError : {avg['AvgAbsoluteError']}")
    print(f"  Avg of MaxAbsoluteError : {avg['MaxAbsoluteError']}")
    print(f"  Avg of MinAbsoluteError : {avg['MinAbsoluteError']}")
