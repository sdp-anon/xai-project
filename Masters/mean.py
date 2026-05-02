import pandas as pd

# Load your CSV
df = pd.read_csv(r"C:\Users\dell.DESKTOP-HTF8KE7\My project\Masters\synthetic_survey_results.csv")
# Select only numeric survey columns
cols = [
    "clarity", "highlight",
    "usefulness", "decision",
    "trust", "reliance",
    "actionable", "independence",
    "effort", "agreement"
]

# Compute mean and std
summary = pd.DataFrame({
    "Mean": df[cols].mean(),
    "Std": df[cols].std()
})

# Round for paper
summary = summary.round(2)

print(summary)

# Convert to LaTeX
latex_table = summary.to_latex()
print(latex_table)