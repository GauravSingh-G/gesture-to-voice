import csv

header = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)] + ['label']

row_a = [0.0]*63 + ['A']
row_b = [0.1]*63 + ['B']

with open('A_data.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerow(row_a)

with open('B_data.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerow(row_b)

print("Created A_data.csv and B_data.csv (sample files).")
