import ast
import os
import subprocess
from typing import Set, Dict
import sys
 
def is_stdlib_module(module_name: str) -> bool:
    """Check if a module is part of Python's standard library."""
    try:
        path = __import__(module_name).__file__
        return 'site-packages' not in path
    except (ImportError, AttributeError, TypeError):
        return True  # Assume built-in modules are stdlib
 
def extract_imports(file_path: str) -> Set[str]:
    """Extract all import names from a Python file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            tree = ast.parse(file.read())
        except SyntaxError:
            print(f"Syntax error in {file_path}")
            return set()
 
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                base_module = name.name.split('.')[0]
                if not is_stdlib_module(base_module):
                    imports.add(base_module)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base_module = node.module.split('.')[0]
                if not is_stdlib_module(base_module):
                    imports.add(base_module)
 
    return imports
 
def scan_directory(directory: str) -> Dict[str, Set[str]]:
    """Recursively scan directory for Python files and their imports."""
    import_locations = {}
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                imports = extract_imports(file_path)
                for imp in imports:
                    if imp not in import_locations:
                        import_locations[imp] = set()
                    import_locations[imp].add(file_path)
    
    return import_locations
 
def add_to_poetry(packages: Dict[str, Set[str]]):
    """Add packages to poetry and handle special cases."""
    special_cases = {
        'cv2': 'opencv-python',
        'PIL': 'pillow',
        'sklearn': 'scikit-learn',
    }
 
    successful = []
    failed = []
 
    for package, locations in packages.items():
        try:
            actual_package = special_cases.get(package, package)
            print(f"\nAttempting to add {actual_package}")
            print(f"Found in: {', '.join(locations)}")
            
            result = subprocess.run(['poetry', 'add', actual_package],
                                 capture_output=True,
                                 text=True)
            
            if result.returncode == 0:
                successful.append(actual_package)
                print(f"✓ Successfully added {actual_package}")
            else:
                failed.append((actual_package, result.stderr))
                print(f"⨯ Failed to add {actual_package}")
                print(f"Error: {result.stderr}")
        
        except Exception as e:
            failed.append((actual_package, str(e)))
            print(f"⨯ Error processing {actual_package}: {str(e)}")
 
    # Print summary
    print("\n=== Summary ===")
    print(f"\nSuccessfully added {len(successful)} packages:")
    for pkg in successful:
        print(f"✓ {pkg}")
 
    if failed:
        print(f"\nFailed to add {len(failed)} packages:")
        for pkg, error in failed:
            print(f"⨯ {pkg}")
            print(f"  Error: {error}")
 
def main():
    if not os.path.exists('pyproject.toml'):
        print("Error: pyproject.toml not found in current directory")
        sys.exit(1)
 
    print("Scanning Python files for imports...")
    import_locations = scan_directory('.')
    
    if not import_locations:
        print("No external imports found")
        return
 
    print("\nFound the following external imports:")
    for package, locations in import_locations.items():
        print(f"{package}:")
        for location in locations:
            print(f"  - {location}")
 
    response = input("\nWould you like to add these packages to poetry? (y/n): ")
    if response.lower() == 'y':
        add_to_poetry(import_locations)
    else:
        print("Operation cancelled")
 
if __name__ == "__main__":
    main()