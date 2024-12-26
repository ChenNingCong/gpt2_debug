import re
import sys
from pathlib import Path

def is_valid_requirement(line):
    """
    Check if a requirement line is valid (package==version format)
    Returns True if valid, False otherwise
    """
    # Remove comments and whitespace
    line = line.split('#')[0].strip()
    
    # Skip empty lines
    if not line:
        return False
    
    # Skip local packages (starting with -e or file:)
    if line.startswith(('-e', 'file:', 'git+')):
        return False
    
    # Check for valid package==version format
    pattern = r'^[a-zA-Z0-9\-_\.]+==\d+(\.\d+)*([a-zA-Z0-9\-_\.]*)?$'
    return bool(re.match(pattern, line))

def clean_requirements(input_file, output_file):
    """
    Clean requirements.txt by removing local packages and keeping only valid package==version lines
    """
    try:
        with open(input_file, 'r') as f:
            requirements = f.readlines()
        
        # Filter valid requirements
        valid_requirements = [
            line.strip()
            for line in requirements
            if is_valid_requirement(line)
        ]
        
        # Write cleaned requirements
        with open(output_file, 'w') as f:
            f.write('\n'.join(valid_requirements))
            f.write('\n')  # Add final newline
        
        print(f"Successfully cleaned requirements. Output written to {output_file}")
        print(f"Removed {len(requirements) - len(valid_requirements)} invalid entries")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python clean_requirements.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'requirements_cleaned.txt'
    
    clean_requirements(input_file, output_file)