#!/usr/bin/env python3
"""Test runner script for MRI classification framework."""

import sys
import pytest
import argparse
from pathlib import Path


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run tests for MRI classification framework")
    parser.add_argument(
        "--unit", 
        action="store_true", 
        help="Run only unit tests"
    )
    parser.add_argument(
        "--integration", 
        action="store_true", 
        help="Run only integration tests"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Run with coverage reporting"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--parallel", "-n", 
        type=int, 
        default=1,
        help="Number of parallel processes (requires pytest-xdist)"
    )
    parser.add_argument(
        "--markers", "-m",
        type=str,
        help="Run tests with specific markers"
    )
    parser.add_argument(
        "test_path",
        nargs="?",
        help="Specific test file or directory to run"
    )
    
    args = parser.parse_args()
    
    # Build pytest arguments
    pytest_args = []
    
    # Determine test paths
    test_dir = Path(__file__).parent
    
    if args.test_path:
        pytest_args.append(args.test_path)
    elif args.unit:
        pytest_args.append(str(test_dir / "unit"))
    elif args.integration:
        pytest_args.append(str(test_dir / "integration"))
    else:
        # Run all tests
        pytest_args.extend([
            str(test_dir / "unit"),
            str(test_dir / "integration")
        ])
    
    # Add verbose output
    if args.verbose:
        pytest_args.append("-v")
    
    # Add coverage
    if args.coverage:
        pytest_args.extend([
            "--cov=mri_node_validator",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Add parallel execution
    if args.parallel > 1:
        pytest_args.extend(["-n", str(args.parallel)])
    
    # Add markers
    if args.markers:
        pytest_args.extend(["-m", args.markers])
    
    # Additional useful options
    pytest_args.extend([
        "--tb=short",  # Shorter traceback format
        "--strict-markers",  # Ensure markers are defined
        "--disable-warnings",  # Reduce noise in output
    ])
    
    print(f"Running tests with arguments: {' '.join(pytest_args)}")
    
    # Run pytest
    exit_code = pytest.main(pytest_args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()