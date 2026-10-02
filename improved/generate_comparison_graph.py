import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure directory exists
os.makedirs('/Users/devrajsinghal/Desktop/capstone/improved/results', exist_ok=True)

# Metrics
metrics = ['Macro F1-Score (%)', 'Calibration Error (ECE) (%)', 'Latency (ms/sample)']

# Values based on the methodology report
# Previous: original paper reported ~96.50% F1, 0.00% ECE (flawed), 0.0033ms latency (flawed batch time)
previous_values = [96.50, 0.00, 0.0033] 
# Improved: 99.29% F1, 0.23% ECE (true), 0.5789 ms latency (true single sample)
improved_values = [99.29, 0.23, 0.5789] 

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle('Comparison: Previous (Flawed) vs. Improved MOO-AutoML Methodology', fontsize=16, fontweight='bold', y=1.05)

labels = ['Previous\n(Reported)', 'Improved\n(Actual)']
colors = ['#ff6666', '#66b3ff']

# 1. Macro F1-Score
axes[0].bar(labels, [previous_values[0], improved_values[0]], color=colors, edgecolor='black')
axes[0].set_title('Macro F1-Score\n(Higher is Better)', pad=15)
axes[0].set_ylabel('Percentage (%)')
axes[0].set_ylim(90, 100)
for i, v in enumerate([previous_values[0], improved_values[0]]):
    axes[0].text(i, v + 0.2, f"{v}%", ha='center', fontweight='bold')

# 2. Calibration Error (ECE)
axes[1].bar(labels, [previous_values[1], improved_values[1]], color=colors, edgecolor='black')
axes[1].set_title('Expected Calibration Error (ECE)\n(Lower is Better)', pad=15)
axes[1].set_ylabel('Percentage (%)')
axes[1].set_ylim(0, 0.3)
axes[1].text(0, previous_values[1] + 0.01, "0.00%\n(Flawed Metric)", ha='center', color='darkred', fontweight='bold')
axes[1].text(1, improved_values[1] + 0.01, f"{improved_values[1]}%\n(True Measurement)", ha='center', fontweight='bold')

# 3. Latency
axes[2].bar(labels, [previous_values[2], improved_values[2]], color=colors, edgecolor='black')
axes[2].set_title('Single-Sample Latency\n(Lower is Better)', pad=15)
axes[2].set_ylabel('Milliseconds (ms)')
axes[2].set_ylim(0, 0.7)
axes[2].text(0, previous_values[2] + 0.02, "0.0033 ms\n(Batch Training Time)", ha='center', color='darkred', fontweight='bold')
axes[2].text(1, improved_values[2] + 0.02, f"{improved_values[2]} ms\n(Single-Sample Time)", ha='center', fontweight='bold')

plt.tight_layout()
output_path = '/Users/devrajsinghal/Desktop/capstone/improved/results/model_comparison.png'
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Graph successfully saved to: {output_path}")
