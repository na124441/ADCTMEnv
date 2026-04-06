"""
Test runner script for the ADCTM project.
This script sets up the local environment specifically for running pytest
and provides an entry point to seamlessly trigger the test suite.
"""
import os
import sys

import pytest


def main() -> int:
    """
    Main execution function for the test suite.
    Configures environment variables for testing, reads command-line arguments,
    and dispatches them to pytest.
    
    Returns:
        int: The exit code returned by pytest (0 for success, non-zero for failure).
    """
    # Disable automatic testing plugins that might interfere with standalone runs
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    
    # Check if arguments were passed via the command line
    # If not, supply default arguments: run quietly (-q) targeting the "tests" directory
    args = sys.argv[1:] or ["-q", "tests"]
    
    # Execute pytest with the specified arguments
    return pytest.main(args)


if __name__ == "__main__":
    # Execute the test runner and exit the system with the returned status code,
    # ensuring CI environments capture test successes/failures correctly.
    raise SystemExit(main())
