import re
import matplotlib.pyplot as plt

# Change this to your log file
file_paths = ["tp1-10.txt", "tp2-10.txt", "tp4-10.txt"]

# Read file

def plot(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Regex to extract TTFT and START
    # pattern = re.compile(r"TTFT:\s*([\d\.]+).*START:\s*([\d\.]+)")
    pattern = re.compile(r"TPOT:\s*([\d\.]+).*START:\s*([\d\.]+)")

    data = []
    for i, line in enumerate(lines):
        match = pattern.search(line)
        if match:
            tpot = float(match.group(1))
            start = float(match.group(2))
            data.append((i, start, tpot))  # (line_index, START, TTFT)

    # Sort by TTFT
    data_sorted = sorted(data, key=lambda x: x[2])

    # Extract for plotting
    tpots = [d[2] for d in data_sorted]
    plt.plot(range(len(tpots)), tpots, label=file_path)

# Plot
plt.figure(figsize=(12,6))

for file_path in file_paths:
    plot(file_path)
plt.xlabel("Prompt ID (line index)")
plt.ylabel("tpots (s)")
plt.tight_layout()

plt.legend()
# plt.show()

plt.savefig("tpot_plot.pdf")
plt.close()
