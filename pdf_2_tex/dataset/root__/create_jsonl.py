# import os
# import json
# import random

# pdf_dir = "/mnt/NAS/patidarritesh/Phoenix_pdfs"
# latex_dir = "/mnt/NAS/patidarritesh/latex_files"

# train_entries = []
# validation_entries = []

# max_train_entries = 100000  # Maximum of 100,000 entries for training
# max_validation_entries = 20000  # Maximum of 20,000 entries for validation

# valid_pairs = []

# latex_files = os.listdir(latex_dir)
# latex_files = latex_files[:120000]  # Consider only the first 120,000 files

# # Read the directory contents once and store in a set for quick lookup
# pdf_files = set(os.listdir(pdf_dir))

# # Loop through the sliced list of LaTeX files
# for latex_file in latex_files:
#     if latex_file.endswith('.tex'):
#         base_name = latex_file[:-4]
#         pdf_file = base_name + '.pdf'

#         if pdf_file in pdf_files:
#             pdf_path = os.path.join(pdf_dir, pdf_file)
#             latex_path = os.path.join(latex_dir, latex_file)
#             valid_pairs.append((pdf_path, latex_path))

# random.shuffle(valid_pairs)

# for pdf_path, latex_path in valid_pairs:
#     if len(train_entries) < max_train_entries:
#         train_entries.append({"pdf": pdf_path, "latex": latex_path})
#     elif len(validation_entries) < max_validation_entries:
#         validation_entries.append({"pdf": pdf_path, "latex": latex_path})
#     else:
#         break

# with open('/mnt/NAS/patidarritesh/root/train.jsonl', 'w') as f:
#     for entry in train_entries:
#         json.dump(entry, f)
#         f.write('\n')

# with open('/mnt/NAS/patidarritesh/root/validation.jsonl', 'w') as f:
#     for entry in validation_entries:
#         json.dump(entry, f)
#         f.write('\n')

# print(f"Created train.jsonl with {len(train_entries)} entries and validation.jsonl with {len(validation_entries)} entries.")


import os
import json
import random
from tqdm import tqdm
import jsonlines

pdf_dir = "/mnt/NAS/patidarritesh/Phoenix_pdfs"
latex_dir = "/mnt/NAS/patidarritesh/Grounding_latex"

train_entries = []
validation_entries = []
test_entries = []
# validation_jsonl_path='/mnt/NAS/patidarritesh/root/validation.jsonl'
max_train_entries = 1600000  # Maximum of 1600,000 entries for training
max_validation_entries = 100000  # Maximum of 10,0000 entries for validation
max_test_entries = 150000  # Maximum of 150,000 entries for testing

valid_pairs = []

latex_files = os.listdir(latex_dir)
print(len(latex_files))
# latex_files = latex_files[:1900000]  # Consider only the first 150,000 files

# Read the directory contents once and store in a set for quick lookup
pdf_files = set(os.listdir(pdf_dir))

# Loop through the sliced list of LaTeX files with a progress bar
for latex_file in tqdm(latex_files, desc="Matching PDFs"):
    if latex_file.endswith('.tex'):
        base_name = latex_file[:-4]
        pdf_file = base_name + '.pdf'

        if pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            latex_path = os.path.join(latex_dir, latex_file)
            valid_pairs.append((pdf_path, latex_path))

random.shuffle(valid_pairs)

# Process the valid pairs and divide them between training and validation
# existing_test_set = set()
# with jsonlines.open('/mnt/NAS/patidarritesh/Pdf_2_LaTeX_v2_LONGFORMER/pdf_2_tex/dataset/root/test.jsonl') as reader:
#     for line in tqdm(reader):
#         existing_pdf_name = line['pdf'].split('/')[-1].replace('.pdf', '.tex')
#         existing_test_set.add(existing_pdf_name)
        
for pdf_path, latex_path in tqdm(valid_pairs, desc="Distributing files"):
    if len(train_entries) < max_train_entries:
        train_entries.append({"pdf": pdf_path, "latex": latex_path})
    elif len(validation_entries) < max_validation_entries:
        validation_entries.append({"pdf": pdf_path, "latex": latex_path})
    elif len(test_entries) < max_test_entries:
        # if pdf_path.split('/')[-1].replace('.pdf', '.tex') not in existing_test_set:
        test_entries.append({"pdf": pdf_path, "latex": latex_path})
    else:
        break  # Stop once the maximum entries are filled

# Write to train.jsonl
with open('/mnt/NAS/patidarritesh/root/train.jsonl', 'w') as f:
    for entry in tqdm(train_entries, desc="Writing train entries"):
        json.dump(entry, f)
        f.write('\n')

# Write to validation.jsonl
with open('/mnt/NAS/patidarritesh/root/validation.jsonl', 'w') as f:
    for entry in tqdm(validation_entries, desc="Writing validation entries"):
        json.dump(entry, f)
        f.write('\n')

with open('/mnt/NAS/patidarritesh/root/test.jsonl', 'w') as f:
    for entry in tqdm(test_entries, desc="Writing test entries"):
        json.dump(entry, f)
        f.write('\n')

print(f"Created train.jsonl with {len(train_entries)} entries and validation.jsonl with {len(validation_entries)} entries and test.jsonl with {len(test_entries)} entries.")
