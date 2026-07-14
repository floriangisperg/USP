"""
Convenience script to run all tests with coverage reporting.
"""
import subprocess
import sys

def run_tests():
    """Run pytest with coverage."""
    print("[!] Running USP test suite...")

    cmd = [
        "pytest",
        "tests/",
        "--cov=src/upstream_analysis",
        "--cov-report=html",
        "--cov-report=term-missing",
        "-v"
    ]

    try:
        result = subprocess.run(cmd, check=True)
        print("[+] All tests passed!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"[-] Tests failed with return code {e.returncode}")
        return e.returncode

if __name__ == "__main__":
    sys.exit(run_tests())
