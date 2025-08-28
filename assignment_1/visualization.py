import re
import matplotlib.pyplot as plt

# Change this to your log file
file_path = "tp1-10.txt"

# Read file
with open(file_path, "r") as f:
    lines = f.readlines()

# Regex to extract TTFT and START
# pattern = re.compile(r"TTFT:\s*([\d\.]+).*START:\s*([\d\.]+)")
pattern = re.compile(r"TPOT:\s*([\d\.]+).*START:\s*([\d\.]+)")

data = []
for i, line in enumerate(lines):
    match = pattern.search(line)
    if match:
        ttpt = float(match.group(1))
        start = float(match.group(2))
        data.append((i, start, ttpt))  # (line_index, START, TTFT)

# Sort by TTFT
data_sorted = sorted(data, key=lambda x: x[2])

# Extract for plotting
prompt_ids = [d[0] for d in data_sorted]
ttpts = [d[2] for d in data_sorted]

# Plot
plt.figure(figsize=(12,6))
plt.bar(range(len(ttpts)), ttpts, tick_label=prompt_ids)
plt.xticks(rotation=90)
plt.xlabel("Prompt ID (line index)")
plt.ylabel("TTPT (s)")
plt.title("Prompt TTPT (sorted)")
plt.tight_layout()
# plt.show()

plt.savefig("ttpt_plot_1.pdf")
plt.close()
