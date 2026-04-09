import os
from collections import Counter
from pathlib import Path

def generate_tree(directory, prefix="", exclude_extensions=None, exclude_folder_contents=None, exclude_folders=None):
    """
    Generate a tree structure excluding specific file types and folder contents.
    
    Args:
        directory: Root directory path
        prefix: Prefix for tree lines (used for recursion)
        exclude_extensions: Set of file extensions to exclude (e.g., {'.jpg', '.png'})
        exclude_folder_contents: Set of folder names whose contents should be hidden
        exclude_folders: Set of folder/file names to completely exclude from the tree
    """
    if exclude_extensions is None:
        exclude_extensions = {'.jpg', '.png', '.jpeg', '.wav', '.bmp'}
    
    if exclude_folder_contents is None:
        exclude_folder_contents = {'train', 'test'}
    
    if exclude_folders is None:
        exclude_folders = set()
    
    try:
        entries = sorted(os.listdir(directory))
    except PermissionError:
        return
    
    # Filter out excluded files/folders and track what was excluded
    filtered_entries = []
    excluded_ext_counts = Counter()  # e.g. {'.bmp': 10, '.jpg': 5}
    for entry in entries:
        if entry in exclude_folders:
            continue
        path = os.path.join(directory, entry)
        if os.path.isfile(path):
            ext = Path(entry).suffix.lower()
            if ext in exclude_extensions:
                excluded_ext_counts[ext] += 1
            else:
                filtered_entries.append(entry)
        else:
            filtered_entries.append(entry)
    
    # Build summary lines for excluded files
    excluded_summaries = []
    for ext in sorted(excluded_ext_counts):
        count = excluded_ext_counts[ext]
        excluded_summaries.append(f"[{count} {ext} file{'s' if count != 1 else ''} excluded]")
    
    # Combine real entries + summary lines for proper tree drawing
    all_items = filtered_entries + excluded_summaries
    
    for i, item in enumerate(all_items):
        is_last = i == len(all_items) - 1
        connector = "└── " if is_last else "├── "
        
        # Summary lines (not real files)
        if i >= len(filtered_entries):
            print(f"{prefix}{connector}{item}")
            continue
        
        entry = item
        path = os.path.join(directory, entry)
        print(f"{prefix}{connector}{entry}")
        
        # If it's a directory, recurse
        if os.path.isdir(path):
            # Check if this folder's contents should be hidden
            if entry in exclude_folder_contents:
                sub_prefix = "    " if is_last else "│   "
                try:
                    count = len(os.listdir(path))
                except PermissionError:
                    count = '?'
                print(f"{prefix}{sub_prefix}└── [{count} items inside, hidden]")
                continue
            
            extension = "    " if is_last else "│   "
            generate_tree(path, prefix + extension, exclude_extensions, exclude_folder_contents, exclude_folders)


if __name__ == "__main__":
    # Configuration
    root_directory = "."  # Current directory
    exclude_extensions = {'.jpg', '.png', '.jpeg', '.wav', '.bmp'}
    exclude_folder_contents = {'train', 'test', '__pycache__', 'node_modules'}
    exclude_folders = {'.git', '.gitattributes', '.gitignore'}
    
    print(root_directory)
    generate_tree(root_directory, "", exclude_extensions, exclude_folder_contents, exclude_folders)
