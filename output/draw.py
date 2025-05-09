import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Read CSV file
df = pd.read_csv('example_run.csv')

# Convert time units from nanoseconds to milliseconds for better display
ns_to_ms = 1e-6
df['arrival_ms'] = df['arrival'] * ns_to_ms
df['end_time_ms'] = df['end_time'] * ns_to_ms
df['queuing_delay_ms'] = df['queuing_delay'] * ns_to_ms
df['TTFT_ms'] = df['TTFT'] * ns_to_ms
df['remaining_time_ms'] = (df['latency'] - df['TTFT']) * ns_to_ms

# Calculate processing start time
df['start_processing_ms'] = df['arrival_ms'] + df['queuing_delay_ms']
df['first_token_ms'] = df['start_processing_ms'] + df['TTFT_ms']

# Set figure size
plt.figure(figsize=(14, 8))

# Create Gantt chart bars for each request
y_ticks = []
y_labels = []

for i, row in df.iterrows():
    # Y-axis position
    y_pos = len(df) - i
    
    # Add to Y-axis labels
    y_ticks.append(y_pos)
    y_labels.append(f"Req {row['request id']} (in:{row['input']}, out:{row['output']})")
    
    # Draw queuing phase (purple)
    plt.barh(y_pos, row['queuing_delay_ms'], left=row['arrival_ms'], 
             color='mediumpurple', alpha=0.6, height=0.5)
    
    # Draw TTFT phase (gold)
    plt.barh(y_pos, row['TTFT_ms'], left=row['start_processing_ms'], 
             color='gold', alpha=0.6, height=0.5)
    
    # Draw remaining token generation phase (darker lavender)
    plt.barh(y_pos, row['remaining_time_ms'], left=row['first_token_ms'], 
             color='darkviolet', alpha=0.6, height=0.5)

# Set Y-axis
plt.yticks(y_ticks, y_labels)

# Set X-axis label to milliseconds
plt.xlabel('Time (milliseconds)')

# Add legend
legend_elements = [
    Patch(facecolor='mediumpurple', alpha=0.6, label='Queuing Time'),
    Patch(facecolor='gold', alpha=0.6, label='First Token Generation (TTFT)'),
    Patch(facecolor='darkviolet', alpha=0.6, label='Remaining Token Generation')
]
plt.legend(handles=legend_elements, loc='upper right')

# Add title
plt.title('Qwen2-MoE Request Processing Time Gantt Chart (gpu_mem = 80)', fontsize=15)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('qwen2_moe_gantt_chart_50req_8block.png', dpi=300)

# Display the figure
plt.show()

print("Gantt chart has been generated and saved as 'qwen2_moe_gantt_chart.png'")
