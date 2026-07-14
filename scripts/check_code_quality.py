"""
Run code quality checks: black, isort, flake8
"""
import subprocess
import sys

def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n[*] {description}...")
    try:
        subprocess.run(cmd, check=True, shell=True)
        print(f"[+] {description} passed")
        return True
    except subprocess.CalledProcessError:
        print(f"[-] {description} failed")
        return False

def main():
    """Run all code quality checks."""
    print("[!] Code Quality Checks")
    print("=" * 40)

    checks = [
        ("black src/ tests/ --check", "Black formatting"),
        ("isort src/ tests/ --check", "Import sorting"),
        ("flake8 src/ tests/ --max-line-length=100", "Flake8 linting")
    ]

    results = [run_command(cmd, desc) for cmd, desc in checks]

    print("\n" + "=" * 40)
    if all(results):
        print("[+] All code quality checks passed!")
        return 0
    else:
        print("[-] Some code quality checks failed")
        print("Run: black src/ tests/ && isort src/ tests/")
        return 1

if __name__ == "__main__":
    sys.exit(main())
