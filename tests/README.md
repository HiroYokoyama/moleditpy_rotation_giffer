# Rotation Giffer — Test Suite

This directory contains the unit tests for the **Rotation Giffer** plugin.

All tests run completely headlessly without requiring an active display/X server or heavy graphical context. The PyQt6 windowing systems and PyVista plotters are dynamically mocked and stubbed at the module level so tests execute in milliseconds.

---

## Running the Tests

To execute the unit test suite, run the following command from the root of the `moleditpy_rotation_giffer` repository:

```bash
python -m pytest tests/ -v
```

To run tests with code coverage analysis:

```bash
python -m pytest tests/ --cov=. --cov-report=term-missing
```

---

## Test Coverage Summary

The test suite covers the following areas in [test_rotation_giffer.py](file:///e:/Research/Calculation/moleditpy/DEV_MAIN/moleditpy_rotation_giffer/tests/test_rotation_giffer.py):

| Class | Coverage / Area |
|---|---|
| `TestGifferMetadata` | Asserts plugin identity constants (`PLUGIN_NAME`, `PLUGIN_VERSION`, `PLUGIN_AUTHOR`, `PLUGIN_CATEGORY`). |
| `TestGifferInitialize` | Asserts that `initialize(context)` registers the correct Export menu action. |
| `TestShowGifferDialog` | Asserts that `show_giffer_dialog` behaves as a singleton (restores/raises existing active dialog and registers new instances safely). |
| `TestOrbitCameraMath` | Validates the Rodrigues orbital camera vector rotation mathematics around the global axes. |
| `TestGifferDialogExecution` | Verifies dialog validation, save path cancellation, standard frame-by-frame loop generation, and direction inversion calculations. |
