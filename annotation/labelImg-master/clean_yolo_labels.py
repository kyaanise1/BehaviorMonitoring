import os
import re

ANNOTATIONS_DIR = r"C:\Users\Jush\BehaviorMonitoring\runs\day9_12_autolabels\labels"

START_FRAME = 11000
END_FRAME = 11719

# Adjusted thresholds targeting x ~ 0.798 and y ~ 0.708
X_CENTER_MIN = 0.78
Y_CENTER_MIN = 0.60

modified_count = 0

for file_name in os.listdir(ANNOTATIONS_DIR):
    if not file_name.endswith('.txt') or file_name == 'classes.txt':
        continue

    numbers = re.findall(r'\d+', file_name)
    if not numbers:
        continue
    
    frame_num = int(numbers[0])

    if START_FRAME <= frame_num <= END_FRAME:
        file_path = os.path.join(ANNOTATIONS_DIR, file_name)
        
        with open(file_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        modified = False

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                x_center = float(parts[1])
                y_center = float(parts[2])

                # Match the right-edge false box position
                if (x_center >= X_CENTER_MIN) and (y_center >= Y_CENTER_MIN):
                    modified = True
                    continue  # Delete this box

            new_lines.append(line)

        if modified:
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            modified_count += 1

print(f"Finished! Cleaned {modified_count} files between {START_FRAME} and {END_FRAME}.")