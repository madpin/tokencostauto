#!/usr/bin/env python3
"""
Script to bump the version in pyproject.toml
"""
import re
import sys
from pathlib import Path


def bump_version(bump_type='patch'):
    """
    Bumps the version in pyproject.toml
    bump_type: 'major', 'minor', 'patch'
    """
    pyproject_path = Path('pyproject.toml')
    if not pyproject_path.exists():
        print("Error: pyproject.toml not found")
        sys.exit(1)
    
    content = pyproject_path.read_text()
    
    # Find the version line
    version_match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if not version_match:
        print("Error: Could not find version in pyproject.toml")
        sys.exit(1)
    
    current_version = version_match.group(1)
    print(f"Current version: {current_version}")
    
    # Split version components
    try:
        major, minor, patch = map(int, current_version.split('.'))
    except ValueError:
        print(f"Error: Invalid version format: {current_version}")
        sys.exit(1)
    
    # Bump according to type
    if bump_type == 'major':
        major += 1
        minor = 0
        patch = 0
    elif bump_type == 'minor':
        minor += 1
        patch = 0
    elif bump_type == 'patch':
        patch += 1
    else:
        print(f"Error: Invalid bump type: {bump_type}")
        sys.exit(1)
    
    new_version = f"{major}.{minor}.{patch}"
    print(f"New version: {new_version}")
    
    # Replace version in content
    new_content = re.sub(
        r'(version\s*=\s*)"([^"]+)"', 
        f'\\1"{new_version}"', 
        content
    )
    
    # Write back to file
    pyproject_path.write_text(new_content)
    print(f"Updated pyproject.toml with version {new_version}")
    
    return new_version


if __name__ == "__main__":
    bump_type = sys.argv[1] if len(sys.argv) > 1 else 'patch'
    bump_version(bump_type)
