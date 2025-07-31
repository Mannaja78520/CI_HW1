with open('Flood_dataset.txt', 'r') as f:
    lines = f.readlines()

# ข้าม 2 บรรทัดแรก (ถ้ามี)
data_lines = lines[2:]

# แปลงข้อมูลเป็นตัวเลขทั้งหมด
data = [list(map(int, line.strip().split())) for line in data_lines]

# คำนวณค่า min และ max ของข้อมูลทั้งหมด
all_values = [val for row in data for val in row]
min_value = min(all_values)
max_value = max(all_values)

print("Minimum value in dataset:", min_value)
print("Maximum value in dataset:", max_value)
