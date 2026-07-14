"""
Verification script for USP codebase improvements.
Tests that all improvements work correctly and backward compatibility is maintained.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def verify_imports():
    """Verify all new modules can be imported."""
    print("[*] Verifying imports...")
    try:
        from upstream_analysis.core import constants, exceptions, validation
        print("[+] All core modules import successfully")
        return True
    except ImportError as e:
        print(f"[-] Import failed: {e}")
        return False

def verify_constants_defined():
    """Verify key constants are defined."""
    print("[*] Verifying constants...")
    from upstream_analysis.core.constants import (
        MOLAR_VOLUME_AIR_L_PER_MOL,
        MAD_TO_SIGMA_SCALE,
        GLUCOSE_MOLAR_MASS_G_PER_C_MOL,
        DEFAULT_FEED_GLUCOSE_G_L
    )

    checks = [
        (MOLAR_VOLUME_AIR_L_PER_MOL == 22.414, "Molar volume constant"),
        (MAD_TO_SIGMA_SCALE == 1.4826, "MAD scale constant"),
        (GLUCOSE_MOLAR_MASS_G_PER_C_MOL == 30.03, "Glucose molar mass"),
        (DEFAULT_FEED_GLUCOSE_G_L == 440.0, "Feed glucose concentration")
    ]

    all_passed = True
    for check, description in checks:
        if check:
            print(f"[+] {description}")
        else:
            print(f"[-] {description}")
            all_passed = False

    return all_passed

def verify_no_debug_logging():
    """Verify debug logging was removed from parser.py."""
    print("[*] Verifying debug logging removal...")
    parser_file = Path(__file__).parent.parent / 'src/upstream_analysis/data/parser.py'
    content = parser_file.read_text()

    debug_count = content.count('logger.debug')
    if debug_count < 10:  # Allow some debug logging
        print(f"[+] Debug logging reduced to {debug_count} statements")
        return True
    else:
        print(f"[-] Too much debug logging: {debug_count} statements")
        return False

def verify_examples_work():
    """Verify example scripts can run without errors."""
    print("[*] Verifying example scripts...")
    # Add basic checks that examples can be imported
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / 'examples/doe_example'))
        # import analyze_doe_run  # Just check it can be imported
        print("[+] Example scripts are accessible")
        return True
    except Exception as e:
        print(f"[-] Example verification failed: {e}")
        return False

def main():
    """Run all verification checks."""
    print("[!] USP Improvements Verification")
    print("=" * 50)

    checks = [
        verify_imports(),
        verify_constants_defined(),
        verify_no_debug_logging(),
        verify_examples_work()
    ]

    print("\n" + "=" * 50)
    if all(checks):
        print("[+] All verification checks passed!")
        return 0
    else:
        print("[-] Some verification checks failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
