import pandas as pd
from transformers import pipeline
import torch

# 1. Load the model from the local directory transferred from your laptop/Colab
classifier = pipeline(
    "zero-shot-classification",
    model="./bart_offline_model",
    device=0 if torch.cuda.is_available() else -1
)

# 2. Define the Labels
SUBJECT_LABELS = [
    "Mathematical Optimization and Convex Programming",
    "Statistics and Applied Machine Learning",
    "Applied Probability and Stochastic Systems",
    "Game Theory and Mechanism Design",
    "Financial Engineering and Quantitative Finance"
]

APPLICATION_LABELS = [
    "Pure Mathematical Theory", "Financial Markets and Asset Pricing",
    "Healthcare Operations and Clinical Decision Support",
    "Natural Language Processing and Text Analytics",
    "Network and Telecommunications Operations",
    "Transportation Networks and Logistics Planning",
    "Supply Chain Management and Revenue Optimization",
    "Robotics and Autonomous Systems", "Energy Systems and Power Grids",
    "Public Policy and Social Operations"
]

# 3. Load Data & Clean
df = pd.read_csv("author_papers_final.csv")

# Ensure text is string, handle any NaNs that slipped in, and truncate to 2000 chars upfront
texts = df['Text'].fillna("No text available").astype(str).str.slice(0, 2000).tolist()

# 4. Batch Inference (The Performance Upgrade)
# Adjust batch_size based on Della's GPU VRAM (16 or 32 is usually safe for BART-large on an A100/V100)
BATCH_SIZE = 16

print("Classifying Subjects (Batch Processing)...")
subject_results = classifier(
    texts,
    SUBJECT_LABELS,
    hypothesis_template="The core academic methodology of this research is {}.",
    multi_label=False,
    batch_size=BATCH_SIZE
)

print("Classifying Applications (Batch Processing)...")
app_results = classifier(
    texts,
    APPLICATION_LABELS,
    hypothesis_template="The primary application domain of this paper is {}.",
    multi_label=False,
    batch_size=BATCH_SIZE
)

# 5. Extract Top 2 Labels and Scores
df['Top_Subjects'] = [res['labels'][:2] for res in subject_results]
df['Subject_Scores'] = [res['scores'][:2] for res in subject_results]

df['Top_Applications'] = [res['labels'][:2] for res in app_results]
df['App_Scores'] = [res['scores'][:2] for res in app_results]

# 6. Aggregate back to the Author Level
def aggregate_labels(series):
    # Flattens the lists of top labels and returns the most frequent ones
    all_labels = [label for sublist in series for label in sublist]
    if not all_labels:
        return []
    return pd.Series(all_labels).value_counts().index[:2].tolist()

print("Aggregating author profiles...")
author_profiles = df.groupby('Author_Id').agg({
    'Top_Subjects': aggregate_labels,
    'Top_Applications': aggregate_labels
}).reset_index()

author_profiles.to_csv("author_classified_profiles.csv", index=False)
print("Classification complete. Profiles saved.")
