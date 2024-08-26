import matplotlib.pyplot as plt
import squarify  # pip install squarify (algorithm for treemap)

# Data for the units and the number of topics in each unit
labels = [
    'Biochemistry and\nCell Biology', 
    'Animal Structure\nand Function', 
    'Plant Structure\nand Function', 
    'Cell Division\nand Principles\nof Genetics', 
    'Evolution', 
    'Biodiversity', 
    'Animal Behavior\nand Principles\nof Ecology'
]
sizes = [
    33, 45, 19, 23, 7, 7, 12  # Number of topics in each unit
]

# Define the colormap
colormap = plt.cm.get_cmap('rainbow', len(sizes))  # 'rainbow' colormap with enough colors

# Create the treemap
plt.figure(figsize=(12, 8))
squarify.plot(sizes=sizes, label=labels, alpha=.8, color=[colormap(i) for i in range(len(sizes))])

# Add title and formatting
plt.title('Biology Map', fontsize=18)
plt.axis('off')  # Remove axis

# Show plot
plt.show()
