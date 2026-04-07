#!/usr/bin/env python3
"""
Project Structure Generator
Mendeteksi struktur folder dan file dalam proyek
"""

import os
import sys
from pathlib import Path

# Folder dan file yang diabaikan
IGNORE_DIRS = {
    '.git',
    '.zig-cache',
    'zig-out',
    'zig-cache',
    '__pycache__',
    'node_modules',
    '.vscode',
    '.idea',
    'target',
    'build',
    'dist',
    'iso_root',
    '.cache',
    'venv',
}

IGNORE_FILES = {
    '.DS_Store',
    'Thumbs.db',
    '.gitkeep',
}

# Ekstensi yang diabaikan
IGNORE_EXTENSIONS = {
    '.o',
    '.obj',
    '.exe',
    '.dll',
    '.so',
    '.dylib',
    '.iso',
    '.img',
    '.bin',
    '.elf',
}

# Deskripsi file (optional)
FILE_DESCRIPTIONS = {
    'main.zig': 'Entry point kernel',
    'vga.zig': 'VGA framebuffer driver',
    'serial.zig': 'Serial port untuk debugging',
    'limine.zig': 'Limine protocol definitions',
    'font.bin': 'Bitmap font 8x16',
    'build.zig': 'Konfigurasi build Zig',
    'linker.ld': 'Linker script',
    'limine.cfg': 'Konfigurasi bootloader',
    'README.md': 'Dokumentasi proyek',
    '.gitignore': 'Git ignore rules',
    
    # Kernel modules
    'gdt.zig': 'Global Descriptor Table',
    'idt.zig': 'Interrupt Descriptor Table',
    'pic.zig': 'Programmable Interrupt Controller',
    'keyboard.zig': 'Keyboard driver',
    'timer.zig': 'Timer/PIT driver',
    'cpu.zig': 'CPU utilities',
    'framebuffer.zig': 'Framebuffer graphics driver',
    'heap.zig': 'Heap memory allocator',
    'paging.zig': 'Virtual memory/paging',
    'pmm.zig': 'Physical memory manager',
    'process.zig': 'Process management',
    'scheduler.zig': 'Process scheduler',
    'switch.zig': 'Context switch',
    'test_procs.zig': 'Test processes',
    
    # Scripts
    'setup-limine.bat': 'Download Limine bootloader',
    'build-iso.bat': 'Buat ISO image',
    'run-qemu.bat': 'Jalankan di QEMU',
    'run-debug.bat': 'Debug dengan GDB',
    'dump_phdr.py': 'ELF header dumper',
}


def should_ignore(name: str, is_dir: bool) -> bool:
    """Check apakah file/folder harus diabaikan"""
    if is_dir:
        return name in IGNORE_DIRS
    
    if name in IGNORE_FILES:
        return True
    
    # Check extension
    ext = os.path.splitext(name)[1].lower()
    if ext in IGNORE_EXTENSIONS:
        return True
    
    return False


def get_description(filename: str) -> str:
    """Ambil deskripsi file jika ada"""
    return FILE_DESCRIPTIONS.get(filename, '')


def generate_tree(
    root_path: str,
    prefix: str = "",
    is_last: bool = True,
    is_root: bool = True,
    show_descriptions: bool = True,
    max_depth: int = -1,
    current_depth: int = 0
) -> list[str]:
    """
    Generate tree structure dari folder
    
    Args:
        root_path: Path ke folder root
        prefix: Prefix untuk indentasi
        is_last: Apakah item terakhir di level ini
        is_root: Apakah ini root folder
        show_descriptions: Tampilkan deskripsi file
        max_depth: Kedalaman maksimum (-1 = unlimited)
        current_depth: Kedalaman saat ini
    
    Returns:
        List of strings representing the tree
    """
    lines = []
    root = Path(root_path)
    
    if not root.exists():
        return [f"Error: Path '{root_path}' tidak ditemukan!"]
    
    # Root folder
    if is_root:
        lines.append(f"{root.name}/")
    
    # Check depth limit
    if max_depth != -1 and current_depth >= max_depth:
        return lines
    
    # Get contents
    try:
        contents = sorted(root.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except PermissionError:
        return lines
    
    # Filter ignored items
    contents = [
        item for item in contents
        if not should_ignore(item.name, item.is_dir())
    ]
    
    # Process each item
    for i, item in enumerate(contents):
        is_last_item = (i == len(contents) - 1)
        
        # Determine connector
        if is_last_item:
            connector = "└── "
            new_prefix = prefix + "    "
        else:
            connector = "├── "
            new_prefix = prefix + "│   "
        
        # Build line
        if item.is_dir():
            line = f"{prefix}{connector}{item.name}/"
            lines.append(line)
            
            # Recurse into directory
            sub_lines = generate_tree(
                str(item),
                new_prefix,
                is_last_item,
                is_root=False,
                show_descriptions=show_descriptions,
                max_depth=max_depth,
                current_depth=current_depth + 1
            )
            lines.extend(sub_lines)
        else:
            # File with optional description
            desc = get_description(item.name) if show_descriptions else ''
            if desc:
                line = f"{prefix}{connector}{item.name:<20} # {desc}"
            else:
                line = f"{prefix}{connector}{item.name}"
            lines.append(line)
    
    return lines


def save_tree(lines: list[str], output_file: str):
    """Simpan tree ke file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Tree saved to: {output_file}")


def print_tree(lines: list[str]):
    """Print tree ke console"""
    for line in lines:
        print(line)


def count_stats(root_path: str) -> dict:
    """Hitung statistik proyek"""
    stats = {
        'total_files': 0,
        'total_dirs': 0,
        'by_extension': {},
        'total_lines': 0,
        'total_size': 0,
    }
    
    root = Path(root_path)
    
    for item in root.rglob('*'):
        # Skip ignored
        skip = False
        for part in item.parts:
            if part in IGNORE_DIRS:
                skip = True
                break
        if skip:
            continue
        
        if item.is_file():
            if should_ignore(item.name, False):
                continue
            
            stats['total_files'] += 1
            stats['total_size'] += item.stat().st_size
            
            ext = item.suffix.lower() or 'no_ext'
            stats['by_extension'][ext] = stats['by_extension'].get(ext, 0) + 1
            
            # Count lines for text files
            if ext in {'.zig', '.py', '.c', '.h', '.rs', '.md', '.txt', '.cfg', '.ld'}:
                try:
                    with open(item, 'r', encoding='utf-8', errors='ignore') as f:
                        stats['total_lines'] += sum(1 for _ in f)
                except:
                    pass
        
        elif item.is_dir():
            if should_ignore(item.name, True):
                continue
            stats['total_dirs'] += 1
    
    return stats


def print_stats(stats: dict):
    """Print statistik proyek"""
    print("\n" + "=" * 50)
    print("PROJECT STATISTICS")
    print("=" * 50)
    print(f"Total Directories: {stats['total_dirs']}")
    print(f"Total Files: {stats['total_files']}")
    print(f"Total Lines of Code: {stats['total_lines']:,}")
    print(f"Total Size: {stats['total_size']:,} bytes ({stats['total_size'] / 1024:.1f} KB)")
    print("\nFiles by Extension:")
    for ext, count in sorted(stats['by_extension'].items(), key=lambda x: -x[1]):
        print(f"  {ext:12} : {count}")
    print("=" * 50)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate project structure tree',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python project_tree.py                    # Current directory
  python project_tree.py /path/to/project   # Specific path
  python project_tree.py -o tree.txt        # Save to file
  python project_tree.py -d 3               # Max depth 3
  python project_tree.py --no-desc          # Without descriptions
  python project_tree.py --stats            # Show statistics
        '''
    )
    
    parser.add_argument(
        'path',
        nargs='?',
        default='.',
        help='Path to project root (default: current directory)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output file to save tree'
    )
    
    parser.add_argument(
        '-d', '--depth',
        type=int,
        default=-1,
        help='Maximum depth (-1 for unlimited)'
    )
    
    parser.add_argument(
        '--no-desc',
        action='store_true',
        help='Hide file descriptions'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show project statistics'
    )
    
    parser.add_argument(
        '--markdown',
        action='store_true',
        help='Wrap output in markdown code block'
    )
    
    args = parser.parse_args()
    
    # Resolve path
    project_path = os.path.abspath(args.path)
    
    if not os.path.exists(project_path):
        print(f"Error: Path '{project_path}' tidak ditemukan!")
        sys.exit(1)
    
    print(f"Scanning: {project_path}\n")
    
    # Generate tree
    lines = generate_tree(
        project_path,
        show_descriptions=not args.no_desc,
        max_depth=args.depth
    )
    
    # Markdown wrapper
    if args.markdown:
        lines = ['```'] + lines + ['```']
    
    # Output
    if args.output:
        save_tree(lines, args.output)
    else:
        print_tree(lines)
    
    # Statistics
    if args.stats:
        stats = count_stats(project_path)
        print_stats(stats)


if __name__ == '__main__':
    main()