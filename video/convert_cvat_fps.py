
import xml.etree.ElementTree as ET
from pathlib import Path

# Original and new frame rates
OLD_FPS = 11.01
NEW_FPS = 15.0

# Input and output annotation files
INPUT_XML = Path("finald12_annotations.xml")
OUTPUT_XML = Path("finald12_annotations_15fps.xml")

# Read original CVAT annotations
tree = ET.parse(INPUT_XML)
root = tree.getroot()

converted = 0
collisions = 0

# Convert frame numbers in CVAT video tracks
for track in root.findall(".//track"):
    used_frames = set()

    for shape in track:
        if "frame" not in shape.attrib:
            continue

        old_frame = int(shape.attrib["frame"])

        # Approximate frame mapping
        new_frame = round(old_frame * NEW_FPS / OLD_FPS)

        if new_frame in used_frames:
            collisions += 1

        used_frames.add(new_frame)
        shape.set("frame", str(new_frame))
        converted += 1

# Save without modifying the original XML
tree.write(
    OUTPUT_XML,
    encoding="utf-8",
    xml_declaration=True
)

print(f"Converted {converted} annotation entries")
print(f"Potential frame collisions: {collisions}")
print(f"Saved: {OUTPUT_XML}")
