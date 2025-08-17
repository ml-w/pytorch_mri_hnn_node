# MRI Classification Framework Test Suite

This directory contains comprehensive unit and integration tests for the MRI classification framework.

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and test configuration
├── run_tests.py                   # Test runner script
├── README.md                      # This file
├── unit/                          # Unit tests
│   ├── test_data/                 # Data loading and processing tests
│   │   ├── test_dataset.py        # MRIDataset, MRIDataModule tests
│   │   ├── test_transforms.py     # 3D transform tests
│   │   └── test_collator.py       # Data collator tests
│   ├── test_models/               # Model component tests
│   │   └── test_hf_detector.py    # HF-compatible model wrapper tests
│   ├── test_evaluation/           # Evaluation component tests
│   │   ├── test_evaluator.py      # Evaluator and metrics tests
│   │   └── test_visualization.py  # 3D visualization tests
│   └── test_training/             # Training pipeline tests
│       └── test_hf_integration.py # HF Trainer integration tests
└── integration/                   # Integration tests
    └── test_end_to_end.py         # Full pipeline integration tests
```

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install pytest pytest-cov pytest-xdist matplotlib seaborn plotly
```

### Basic Usage

Run all tests:
```bash
cd /path/to/pytorch_mri_hnn_node
python -m pytest mri_node_validator/tests/
```

Or use the test runner:
```bash
python mri_node_validator/tests/run_tests.py
```

### Test Categories

**Unit Tests Only:**
```bash
python mri_node_validator/tests/run_tests.py --unit
```

**Integration Tests Only:**
```bash
python mri_node_validator/tests/run_tests.py --integration
```

**With Coverage:**
```bash
python mri_node_validator/tests/run_tests.py --coverage
```

**Verbose Output:**
```bash
python mri_node_validator/tests/run_tests.py --verbose
```

**Parallel Execution:**
```bash
python mri_node_validator/tests/run_tests.py --parallel 4
```

**Specific Test Categories:**
```bash
python mri_node_validator/tests/run_tests.py --markers "visualization"
python mri_node_validator/tests/run_tests.py --markers "model"
```

### Running Specific Tests

**Single test file:**
```bash
python -m pytest mri_node_validator/tests/unit/test_data/test_dataset.py
```

**Single test class:**
```bash
python -m pytest mri_node_validator/tests/unit/test_data/test_dataset.py::TestMRIDataset
```

**Single test method:**
```bash
python -m pytest mri_node_validator/tests/unit/test_data/test_dataset.py::TestMRIDataset::test_initialization
```

## Test Coverage

The test suite aims for comprehensive coverage:

| Component | Target Coverage | Description |
|-----------|----------------|-------------|
| Data Loading | 95%+ | Dataset, transforms, collators |
| Model Architecture | 90%+ | HF wrapper, forward pass, configuration |
| HF Integration | 95%+ | Trainer compatibility, loss computation |
| Evaluation/Metrics | 95%+ | Evaluator, metrics, visualization |
| Visualization | 85%+ | 3D slice extraction, plotting |
| Training Pipeline | 90%+ | End-to-end training workflow |

## Key Test Features

### 3D Volume Support
- Tests verify 3D tensor shapes (C, D, H, W) throughout pipeline
- Slice extraction for visualization components
- Memory efficiency with large volumes

### HF Trainer Compatibility
- Model wrapper tests for transformers integration
- Data collator tests for proper batching
- Loss computation for binary classification
- Custom compute_metrics function

### Realistic Data Simulation
- Synthetic NIfTI file generation
- CSV label file creation
- Configuration file testing
- Temporary directory management

### Error Handling
- Malformed data scenarios
- Missing files and labels
- Invalid configurations
- Memory constraints

## Test Fixtures

Key fixtures available in `conftest.py`:

- `sample_mri_volume`: 3D MRI tensor (1, 64, 64, 64)
- `sample_nifti_files`: Realistic NIfTI files in temp directory
- `sample_csv_file`: CSV with patient labels
- `training_config`: Complete training configuration
- `mock_hf_model_config`: HF model configuration
- `sample_predictions`: Model prediction results
- `temp_dir`: Temporary directory for file operations

## Performance Tests

Integration tests include performance benchmarks:

- **Inference Speed**: Samples processed per second
- **Memory Usage**: Memory consumption tracking
- **Large Volume Handling**: 128³ voxel processing
- **Reproducibility**: Consistent results with seeds

## Continuous Integration

Tests are designed for CI environments:

- Non-interactive matplotlib backend
- CPU-only execution (GPU tests marked separately)
- Reasonable memory requirements
- Fast execution times
- Comprehensive error reporting

## Adding New Tests

### Unit Test Template

```python
import pytest
import torch
from mri_node_validator.your_module import YourClass

class TestYourClass:
    def test_initialization(self):
        \"\"\"Test class initialization.\"\"\"
        obj = YourClass()
        assert obj is not None
    
    def test_method_with_fixture(self, sample_mri_volume):
        \"\"\"Test method using fixture.\"\"\"
        obj = YourClass()
        result = obj.process(sample_mri_volume)
        assert result.shape == sample_mri_volume.shape
```

### Integration Test Template

```python
import pytest
from mri_node_validator.your_module import complete_pipeline

class TestYourPipeline:
    def test_end_to_end_workflow(self, temp_dir, sample_data):
        \"\"\"Test complete workflow.\"\"\"
        result = complete_pipeline(
            data_dir=temp_dir,
            config=sample_data
        )
        assert result['success'] == True
```

### Test Markers

Mark tests for categorization:

```python
@pytest.mark.unit
@pytest.mark.model
def test_model_function():
    pass

@pytest.mark.integration
@pytest.mark.slow
def test_training_pipeline():
    pass

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_gpu_training():
    pass
```

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Run from repository root
cd /path/to/pytorch_mri_hnn_node
export PYTHONPATH=$PYTHONPATH:$(pwd)
python -m pytest mri_node_validator/tests/
```

**Memory Issues:**
```bash
# Run with smaller batch sizes or fewer parallel processes
python mri_node_validator/tests/run_tests.py --parallel 1
```

**Matplotlib Issues:**
```bash
# Install GUI backend or use headless mode
export MPLBACKEND=Agg
python -m pytest mri_node_validator/tests/
```

**Missing Dependencies:**
```bash
# Install all test dependencies
pip install -r mri_node_validator/requirements.txt
pip install pytest pytest-cov pytest-xdist
```

### Debug Mode

Run with debugging:
```bash
python -m pytest mri_node_validator/tests/ --pdb --tb=long -v
```

### Specific Environment Issues

**Docker/CI:**
```bash
# Use headless mode
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 &
python -m pytest mri_node_validator/tests/
```

**Resource Constraints:**
```bash
# Limit parallel processes and memory usage
python -m pytest mri_node_validator/tests/ --maxfail=1 -x
```

## Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Ensure 90%+ code coverage for new components
3. Include both unit and integration tests
4. Add appropriate test markers
5. Update this documentation if needed
6. Run full test suite before submitting

For questions or issues with the test suite, please refer to the main project documentation or submit an issue.