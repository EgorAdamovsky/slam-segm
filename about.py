import os
import sys
from pathlib import Path
from datetime import datetime

# Директории и файлы, которые следует игнорировать
EXCLUDE_DIRS = {
    '.git', '__pycache__', 'venv', 'env', '.venv', 
    'checkpoints', 'data', 'runs', 'logs', 'node_modules',
    '.idea', '.vscode', 'build', 'dist', '.pytest_cache',
    'wandb', 'mlruns', 'outputs'
}

EXCLUDE_EXTENSIONS = {
    '.pyc', '.pyo', '.pth', '.pt', '.bin', '.onnx',
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff',
    '.mp4', '.avi', '.mov', '.mkv',
    '.csv', '.npy', '.npz', '.h5', '.hdf5',
    '.pdf', '.doc', '.docx', '.xls', '.xlsx',
    '.zip', '.tar', '.gz', '.rar', '.7z',
    '.exe', '.dll', '.so', '.dylib',
    '.db', '.sqlite', '.sqlite3',
    '.log', '.lock', '.DS_Store', '.gitignore'
}

OUTPUT_FILE = "dump.md"

def should_exclude(path: Path) -> bool:
    """Проверяет, должен ли файл/директория быть исключен."""
    # Проверяем директории в пути
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
    
    # Проверяем расширение файла
    if path.suffix.lower() in EXCLUDE_EXTENSIONS:
        return True
    
    # Исключаем сам скрипт дампа и выходной файл
    if path.name in {'about.py', 'dump_code.py', OUTPUT_FILE}:
        return True
    
    return False

def detect_encoding(file_path: Path) -> str:
    """Пытается определить кодировку файла."""
    encodings = ['utf-8', 'cp1251', 'latin-1', 'ascii']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read()
            return encoding
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    return 'utf-8'  # Fallback

def get_language_from_extension(ext: str) -> str:
    """Возвращает язык для подсветки синтаксиса Markdown."""
    lang_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'cpp',
        '.hpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.java': 'java',
        '.sh': 'bash',
        '.bash': 'bash',
        '.bat': 'batch',
        '.cmd': 'batch',
        '.ps1': 'powershell',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.json': 'json',
        '.xml': 'xml',
        '.html': 'html',
        '.css': 'css',
        '.sql': 'sql',
        '.md': 'markdown',
        '.txt': 'text',
        '.toml': 'toml',
        '.ini': 'ini',
        '.cfg': 'ini',
        '.dockerfile': 'dockerfile',
        '.r': 'r',
        '.R': 'r',
        '.jl': 'julia',
        '.rs': 'rust',
        '.go': 'go',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.cmake': 'cmake',
        '.makefile': 'makefile',
    }
    return lang_map.get(ext.lower(), 'text')

def main():
    root_dir = Path('.')
    
    print(f"Начинаю сканирование директории: {root_dir.absolute()}")
    print(f"Исключаю директории: {', '.join(EXCLUDE_DIRS)}")
    print(f"Исключаю расширения: {', '.join(sorted(EXCLUDE_EXTENSIONS))}")
    print("-" * 60)
    
    # Собираем все файлы
    all_files = []
    for file_path in root_dir.rglob('*'):
        if file_path.is_file() and not should_exclude(file_path):
            all_files.append(file_path)
    
    all_files.sort()
    
    print(f"Найдено файлов для обработки: {len(all_files)}")
    print("-" * 60)
    
    # Записываем в выходной файл
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        out_f.write("# Полный дамп кода репозитория ReturnRoute\n\n")
        out_f.write(f"**Дата генерации:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        out_f.write(f"**Количество файлов:** {len(all_files)}\n\n")
        out_f.write("---\n\n")
        
        processed_count = 0
        skipped_count = 0
        
        for file_path in all_files:
            try:
                # Определяем кодировку
                encoding = detect_encoding(file_path)
                
                # Читаем содержимое
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read()
                
                # Пропускаем пустые файлы
                if not content.strip():
                    skipped_count += 1
                    continue
                
                # Получаем язык для подсветки
                language = get_language_from_extension(file_path.suffix)
                
                # Записываем в Markdown
                out_f.write(f"## Файл: `{file_path.as_posix()}`\n\n")
                out_f.write(f"```{language}\n")
                out_f.write(content)
                if not content.endswith('\n'):
                    out_f.write('\n')
                out_f.write("```\n\n")
                
                processed_count += 1
                
                # Прогресс
                if processed_count % 10 == 0:
                    print(f"Обработано файлов: {processed_count}/{len(all_files)}")
                    
            except Exception as e:
                print(f"Ошибка при обработке {file_path}: {e}")
                skipped_count += 1
                continue
    
    print("-" * 60)
    print(f"✓ Готово!")
    print(f"✓ Обработано файлов: {processed_count}")
    print(f"✓ Пропущено файлов: {skipped_count}")
    print(f"✓ Выходной файл: {OUTPUT_FILE}")
    
    # Показываем размер файла
    file_size = os.path.getsize(OUTPUT_FILE)
    if file_size < 1024:
        size_str = f"{file_size} B"
    elif file_size < 1024 * 1024:
        size_str = f"{file_size / 1024:.2f} KB"
    else:
        size_str = f"{file_size / (1024 * 1024):.2f} MB"
    
    print(f"✓ Размер файла: {size_str}")
    print("-" * 60)

if __name__ == "__main__":
    main()