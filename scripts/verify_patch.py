#!/usr/bin/env python3
"""OKC Serving Track Patch Validator.

Validates patch.diff against all official constraints defined in serving/RULES.md:
1. File size must be <= 400 KB (409,600 bytes).
2. No added line may exceed 400 characters.
3. Changed files must be strictly under 'python/sglang/'.
4. Must NOT touch weight loading ('python/sglang/srt/model_loader/').
5. Only allowed source file extensions (.py, .cu, .cuh, .h, .hpp, .c, .cc, .cpp, .json).
6. No file renames, mode changes, or binary diffs.
7. Python syntax check: all modified .py files compile cleanly.
"""

import sys
import os
import re
import py_compile

MAX_PATCH_BYTES = 400 * 1024  # 400 KB
MAX_LINE_CHARS = 400
ALLOWED_EXTENSIONS = {'.py', '.cu', '.cuh', '.h', '.hpp', '.c', '.cc', '.cpp', '.json'}
DISALLOWED_PREFIXES = ['python/sglang/srt/model_loader/']
REQUIRED_PREFIX = 'python/sglang/'

def validate_patch(patch_path: str, repo_root: str = None) -> bool:
    if not os.path.exists(patch_path):
        print(f"❌ ERROR: Patch file not found: {patch_path}")
        return False

    file_size = os.path.getsize(patch_path)
    print(f"🔍 Validating patch: {patch_path}")
    print(f"   Size: {file_size:,} bytes / {MAX_PATCH_BYTES:,} bytes max ({file_size / MAX_PATCH_BYTES * 100:.1f}%)")

    errors = []
    warnings = []

    # 1. Check size limit
    if file_size > MAX_PATCH_BYTES:
        errors.append(f"Patch size ({file_size:,} bytes) exceeds 400 KB limit ({MAX_PATCH_BYTES:,} bytes).")

    with open(patch_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    current_file = None
    files_touched = set()

    for line_idx, line in enumerate(lines, 1):
        line_clean = line.rstrip('\r\n')

        # Check for binary diffs
        if line_clean.startswith('Binary files'):
            errors.append(f"Line {line_idx}: Binary diff detected: {line_clean}")

        # Check for file renames or mode changes
        if line_clean.startswith('rename from') or line_clean.startswith('rename to'):
            errors.append(f"Line {line_idx}: File rename detected (disallowed): {line_clean}")
        if line_clean.startswith('old mode') or line_clean.startswith('new mode'):
            errors.append(f"Line {line_idx}: File mode change detected (disallowed): {line_clean}")

        # Parse diff header
        if line_clean.startswith('diff --git'):
            match = re.search(r'diff --git a/(.*?) b/(.*)', line_clean)
            if match:
                file_b = match.group(2)
                current_file = file_b
                files_touched.add(file_b)

                # Validate required prefix
                if not file_b.startswith(REQUIRED_PREFIX):
                    errors.append(f"Line {line_idx}: File '{file_b}' is outside required prefix '{REQUIRED_PREFIX}'.")

                # Validate disallowed prefix
                for dis in DISALLOWED_PREFIXES:
                    if file_b.startswith(dis):
                        errors.append(f"Line {line_idx}: File '{file_b}' modifies forbidden model loader '{dis}'.")

                # Validate file extension
                _, ext = os.path.splitext(file_b)
                if ext not in ALLOWED_EXTENSIONS:
                    errors.append(f"Line {line_idx}: File '{file_b}' has disallowed extension '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")

        # Check added line length (starts with '+', but not '+++')
        if line.startswith('+') and not line.startswith('+++'):
            # The '+' marker doesn't count against line length on the applied file, but check both
            added_text = line[1:].rstrip('\r\n')
            if len(added_text) > MAX_LINE_CHARS:
                errors.append(
                    f"Line {line_idx} in {current_file or 'unknown'}: Added line has {len(added_text)} characters "
                    f"(exceeds {MAX_LINE_CHARS} char limit):\n      '{added_text[:80]}...'"
                )

    # 7. Check Python syntax for modified .py files
    if repo_root:
        for rel_file in files_touched:
            if rel_file.endswith('.py'):
                full_path = os.path.join(repo_root, rel_file)
                if os.path.exists(full_path):
                    try:
                        py_compile.compile(full_path, doraise=True)
                    except py_compile.PyCompileError as e:
                        errors.append(f"Syntax error in modified file {rel_file}: {e}")

    print(f"\n📂 Files touched ({len(files_touched)}):")
    for f in sorted(files_touched):
        print(f"   • {f}")

    if errors:
        print(f"\n❌ FAILED: {len(errors)} violation(s) found:")
        for err in errors:
            print(f"   • {err}")
        return False
    else:
        print("\n✅ PASSED: Patch strictly complies with all serving/RULES.md constraints!")
        return True

if __name__ == '__main__':
    default_patch = os.path.join(os.path.dirname(__file__), '..', 'serving', 'patch.diff')
    patch_file = sys.argv[1] if len(sys.argv) > 1 else default_patch
    
    # Try finding SGLang repo root
    sglang_root = os.path.join(os.path.dirname(__file__), '..', 'serving', 'sglang')
    root = sglang_root if os.path.isdir(sglang_root) else None

    success = validate_patch(patch_file, root)
    sys.exit(0 if success else 1)
