from fetch_data import *
import pandas as pd
import matplotlib.pyplot as plt


data = get_data_as_list(type="strings")

# Check unique chord frequency 

frequencies = {}

for song in data:
    uniq_song = set(song)
    # Got the unique chords in the song
    num_unique_chords = len(uniq_song)

    if num_unique_chords in frequencies:
        frequencies[num_unique_chords] += 1
    else:
        frequencies[num_unique_chords] = 1

df = pd.DataFrame(
    list(frequencies.items()), 
    columns=['Number of Unique Chords', 'Frequency']
)
df = df.sort_values(by='Number of Unique Chords')

plt.figure(figsize=(14, 7))
plt.bar(df['Number of Unique Chords'], df['Frequency'], alpha=0.9)
plt.xlabel('Number of Unique Chords', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Frequency Distribution of Unique Chords per Song', fontsize=14, fontweight='bold')

# Ensure x-axis ticks are readable
plt.xticks(df['Number of Unique Chords'][::2], rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
