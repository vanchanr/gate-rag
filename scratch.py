import os
import shutil

subjects = ["CE", "CH", "CS", "DA", "EC", "EE", "MA", "ME"]

for subject in subjects:
    src_path = f"GATE _{subject}_2025_Syllabus.pdf"
    dst_path = f"data/pdf/{subject}/syllabus.pdf"
    shutil.move(src_path, dst_path)
