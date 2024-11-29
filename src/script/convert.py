import pandas as pd
import os
from pathlib import Path

def convert_excel_to_csv(excel_path):
    try:
        # Read Excel file
        df = pd.read_excel(excel_path)
        
        # Create output filename
        output_path = Path(str(excel_path).replace('.xlsx', '.csv'))
        
        # Save as CSV with UTF-8 encoding
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully converted: {excel_path} -> {output_path}")
        
    except Exception as e:
        print(f"Error processing {excel_path}: {str(e)}")

# List of files to process
files = [
    "/home/dukhanin/soc_hack/docs/Хакатон_2024/Примеры нерелевантных постов.xlsx",
    "/home/dukhanin/soc_hack/docs/Хакатон_2024/Примеры релевантных постов_2.xlsx",
    "docs/Хакатон_2024/примеры релевантных постов.xlsx"
]

# Process each file
for file_path in files:
    if os.path.exists(file_path):
        convert_excel_to_csv(file_path)
    else:
        print(f"File not found: {file_path}")