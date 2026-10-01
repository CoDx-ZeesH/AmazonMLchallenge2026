import os
import zipfile
import argparse
import sys

def validate_and_package(team_name):
    output_dir = 'output'
    required_files = [
        f'{output_dir}/matching_results.tsv',
        f'{output_dir}/candidate_pairs.tsv',
        'Documentation_template.md'
    ]
    
    print(f"Preparing package for team: {team_name}")
    print("-" * 40)
    print("Step 1: Validating required files...")
    
    for filepath in required_files:
        if not os.path.exists(filepath):
            print(f"[ERROR] Missing required file: {filepath}")
            print("Please run the pipeline successfully and fill out the documentation before packaging.")
            sys.exit(1)
            
    print("Validation passed. All required files found.")
    print("-" * 40)
    print("Step 2: Creating submission zip archive...")
    
    zip_name = f"{team_name}_submission.zip"
    
    # We need to zip output/, code/ and Documentation_template.md
    target_dirs = ['output', 'code', 'utils']
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add documentation
        zipf.write('Documentation_template.md', 'Documentation_template.md')
        
        # Add target directories
        for target in target_dirs:
            if os.path.exists(target):
                for root, dirs, files in os.walk(target):
                    if '__pycache__' in root:
                        continue
                    for file in files:
                        if file.endswith('.pyc'):
                            continue
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, file_path)
                
    print("-" * 40)
    print(f"[SUCCESS] Packaged submission into {zip_name}")
    print("You can now submit this zip file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package project for submission.")
    parser.add_argument('--team', type=str, required=True, help="Your team name without spaces (e.g. DataNinjas)")
    args = parser.parse_args()
    validate_and_package(args.team)
