import os
import re

def modify_files_in_directory(directory_path):
    # Pattern to match files starting with 's1'
    file_pattern = re.compile(r'^s3.*\.cali$')

    for filename in os.listdir(directory_path):
        if file_pattern.match(filename):
            # Construct full file path
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                content = file.read()
            
            # Replace 'Random' with 'Sorted'
            modified_content = content.replace('Random', '1%%perturbed')

            with open(file_path, 'w') as file:
                file.write(modified_content)
                
modify_files_in_directory('.')