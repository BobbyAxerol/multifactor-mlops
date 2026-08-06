"""
Automated Repository Codebase Integrity Test.
Scans all Python files across the codebase to enforce zero prohibited .bfill() usage.
"""

import os
import glob
import pytest

def test_no_prohibited_bfill_in_entire_codebase():
    """
    Scans all Python source files in src/
    Asserts zero occurrence of .bfill( or method='bfill' or method="bfill".
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    search_dirs = [
        os.path.join(root_dir, "src")
    ]

    violations = []

    for s_dir in search_dirs:
        if not os.path.exists(s_dir):
            continue
        for root, _, files in os.walk(s_dir):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    with open(filepath, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        for line_num, line in enumerate(lines, 1):
                            if ".bfill(" in line or "method='bfill'" in line or 'method="bfill"' in line:
                                violations.append(f"{os.path.relpath(filepath, root_dir)}:L{line_num}: {line.strip()}")

    assert len(violations) == 0, f"Prohibited .bfill() usage found in codebase:\n" + "\n".join(violations)
