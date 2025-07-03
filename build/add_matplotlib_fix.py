#!/usr/bin/env python3
"""
Script to add matplotlib backend fix to inference.py
"""

# The fix to add at the beginning of inference.py
MATPLOTLIB_FIX = '''# Force matplotlib to use Agg backend before any imports
import os
os.environ['MPLBACKEND'] = 'Agg'

'''

def add_fix_to_file(filename):
    """Add the matplotlib fix to the beginning of a Python file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if fix already exists
        if "os.environ['MPLBACKEND']" in content:
            print(f"✓ Matplotlib fix already present in {filename}")
            return
        
        # Find the first import statement
        lines = content.split('\n')
        insert_index = 0
        
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                insert_index = i
                break
        
        # Insert the fix before the first import
        lines.insert(insert_index, MATPLOTLIB_FIX.strip())
        
        # Write back
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"✓ Added matplotlib fix to {filename}")
        
    except Exception as e:
        print(f"✗ Error processing {filename}: {e}")

if __name__ == "__main__":
    add_fix_to_file("inference.py")
    add_fix_to_file("inference_ray.py")
    add_fix_to_file("inference_ray_optimized.py")