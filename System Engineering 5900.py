#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import os
import json

# Set seed for reproducibility
np.random.seed(42)
N_TOTAL = 1000 # Pilot size as specified (800 train, 100 dev, 100 test)

# 1. Generate Mock "Official" EmoBank Data (1-5 scale)
emotions = ['joy', 'fear', 'sadness', 'anger', 'surprise', 'neutral']
contexts = ['Twitter', 'Blogs', 'Fiction', 'News']

data = {
    'sequence_id': [f"seq_{i:04d}" for i in range(N_TOTAL)],
    'text': [f"This is a sample sentence representing {np.random.choice(emotions)} in a {np.random.choice(contexts)} context." for i in range(N_TOTAL)],
    'emotion': np.random.choice(emotions, N_TOTAL),
    # Official scores are typically 1.0 to 5.0
    'V_official': np.random.uniform(1.0, 5.0, N_TOTAL),
    'A_official': np.random.uniform(1.0, 5.0, N_TOTAL),
    'D_official': np.random.uniform(1.0, 5.0, N_TOTAL),
}

df_official = pd.DataFrame(data)

# 2. Apply the specified transformation: normalized = (official_score - 3) / 2
df_manifold = df_official.copy()
df_manifold['valence'] = (df_official['V_official'] - 3) / 2
df_manifold['arousal'] = (df_official['A_official'] - 3) / 2
df_manifold['dominance'] = (df_official['D_official'] - 3) / 2

# Drop the official columns for the manifold-ready dataset
df_manifold = df_manifold.drop(columns=['V_official', 'A_official', 'D_official'])

# 3. Create Directory Structure
os.makedirs('data/emobank', exist_ok=True)

# 4. Save Full Manifold Dataset
df_manifold.to_csv('data/emobank/emobank_manifold.csv', index=False)

# 5. Create Pilot Split (800 train, 100 dev, 100 test)
# Shuffle deterministically
df_shuffled = df_manifold.sample(frac=1, random_state=42).reset_index(drop=True)

train_df = df_shuffled.iloc[:800]
dev_df = df_shuffled.iloc[800:900]
test_df = df_shuffled.iloc[900:1000]

train_df.to_csv('data/emobank/emobank_pilot_train.csv', index=False)
dev_df.to_csv('data/emobank/emobank_pilot_dev.csv', index=False)
test_df.to_csv('data/emobank/emobank_pilot_test.csv', index=False)

# Combine into a single pilot file with a 'split' column for convenience
df_pilot = pd.concat([
    train_df.assign(split='train'),
    dev_df.assign(split='dev'),
    test_df.assign(split='test')
])
df_pilot.to_csv('data/emobank/emobank_pilot_1000.csv', index=False)

# 6. Generate Metadata
metadata = {
    "source": "EmoBank (Mocked for Pipeline Testing)",
    "license": "CC BY-SA 4.0",
    "total_usable_sentences": len(df_manifold),
    "pilot_split_counts": {"train": 800, "dev": 100, "test": 100},
    "score_transformation": "normalized = (official_score - 3) / 2",
    "original_scale": "1.0 to 5.0",
    "normalized_scale": "-1.0 to 1.0",
    "emotions_present": emotions
}

with open('data/emobank/dataset_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print("✅ Successfully generated mock EmoBank pipeline data in `data/emobank/`")
print(f"   - emobank_manifold.csv: {len(df_manifold)} rows")
print(f"   - emobank_pilot_1000.csv: {len(df_pilot)} rows (800/100/100 split)")


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the generated pilot data
df = pd.read_csv('data/emobank/emobank_pilot_1000.csv')

# Set up the plot style
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 8))

# Define a color palette for the emotions
emotion_palette = {
    'joy': '#FFD700',       # Gold
    'fear': '#800080',      # Purple
    'sadness': '#00008B',   # Dark Blue
    'anger': '#8B0000',     # Dark Red
    'surprise': '#FF8C00',  # Dark Orange
    'neutral': '#808080'    # Gray
}

# Scatter plot of Valence vs Arousal
scatter = sns.scatterplot(
    data=df,
    x='valence',
    y='arousal',
    hue='emotion',
    palette=emotion_palette,
    s=60,
    alpha=0.7,
    edgecolor='black',
    linewidth=0.5,
    ax=ax
)

# Add center crosshairs (0,0)
ax.axhline(0, color='black', linewidth=1, linestyle='--')
ax.axvline(0, color='black', linewidth=1, linestyle='--')

# Formatting
ax.set_title('Control: Valence-Arousal Distribution of EmoBank Pilot (Pre-Isomap)', fontsize=14, fontweight='bold')
ax.set_xlabel('Valence (Negative ← 0 → Positive)', fontsize=12)
ax.set_ylabel('Arousal (Calm ← 0 → Activated)', fontsize=12)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)

# Move legend outside
plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('va_circumplex_control.png', dpi=300, bbox_inches='tight')
print("✅ Saved Valence-Arousal control visualization to 'va_circumplex_control.png'")


# In[6]:


pip install torch transformers scikit-learn pandas numpy matplotlib seaborn


# In[ ]:




