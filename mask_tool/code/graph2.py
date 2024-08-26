import json
import matplotlib.pyplot as plt
import squarify  # pip install squarify (algorithm for treemap)

# Function to truncate long labels
def truncate_label(label, max_length=20):
    return (label[:max_length] + '...') if len(label) > max_length else label

# Load data from JSON file
with open('mask_tool\\code\\biology_data.json', 'r') as file:
    units = json.load(file)

# Create treemaps for each unit
fig, axs = plt.subplots(len(units), 1, figsize=(12, len(units) * 6), sharex=True)

for ax, (unit, topics) in zip(axs, units.items()):
    truncated_labels = [truncate_label(topic) for topic in topics]
    sizes = [1] * len(topics)  # Each topic gets an equal share
    squarify.plot(sizes=sizes, label=truncated_labels, alpha=.8, color=plt.cm.tab20.colors, ax=ax)
    ax.set_title(unit, fontsize=16)
    ax.axis('off')  # Remove axis

plt.tight_layout()
plt.show()
