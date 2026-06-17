"""
Integration tests for rotation_giffer.py
Verifies the plugin contract against a stub PluginContext without Qt/PyVista.

Two execution modes
-------------------
1. **Stub mode** (always runs, including CI):
   A _StubContext mirrors the real PluginContext API so all contract tests
   pass without installing the main app.

2. **Real-context mode** (optional):
   If python_molecular_editor/moleditpy/src is found relative to this repo
   (local dev, sibling directory) OR via CI_MAIN_APP_SRC env var, the tests
   are also run using the actual PluginContext class.
   Skipped when the main app is not available.

CI setup
--------
Add a step to the workflow before running pytest:

    - name: Clone main app (for real-context integration tests)
      run: git clone --depth 1 https://github.com/HiroYokoyama/python_molecular_editor.git
             ../python_molecular_editor || true
"""
import sys
import os
import types
import unittest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub Qt modules before importing the plugin
# ---------------------------------------------------------------------------

def _install_qt_stubs():
    if "PyQt6" in sys.modules and hasattr(sys.modules["PyQt6"], "__file__"):
        return  # real Qt already present

    pyqt6 = types.ModuleType("PyQt6")
    qt_core = types.ModuleType("PyQt6.QtCore")

    class _Qt:
        class AlignmentFlag:
            AlignRight = 0
        class Orientation:
            Horizontal = 0

    qt_core.Qt = _Qt
    qt_core.QCoreApplication = MagicMock()

    qt_widgets = types.ModuleType("PyQt6.QtWidgets")
    for cls_name in [
        "QDialog", "QVBoxLayout", "QHBoxLayout", "QComboBox", "QSpinBox",
        "QCheckBox", "QPushButton", "QFileDialog", "QMessageBox", "QFormLayout",
    ]:
        setattr(qt_widgets, cls_name, MagicMock())

    qt_gui = types.ModuleType("PyQt6.QtGui")

    for name, mod in [
        ("PyQt6", pyqt6),
        ("PyQt6.QtCore", qt_core),
        ("PyQt6.QtWidgets", qt_widgets),
        ("PyQt6.QtGui", qt_gui),
    ]:
        sys.modules.setdefault(name, mod)

    sys.modules.setdefault("PIL", types.ModuleType("PIL"))
    sys.modules.setdefault("PIL.Image", types.ModuleType("PIL.Image"))


_install_qt_stubs()

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

import rotation_giffer as _pkg
from rotation_giffer import initialize, PLUGIN_NAME, PLUGIN_VERSION


# ---------------------------------------------------------------------------
# Stub PluginContext
# ---------------------------------------------------------------------------

class _StubContext:
    def __init__(self):
        self._export_actions = []
        self._windows = {}

    def add_export_action(self, label, callback):
        self._export_actions.append((label, callback))

    def get_window(self, key):
        return self._windows.get(key)

    def register_window(self, key, win):
        self._windows[key] = win

    def get_main_window(self):
        return MagicMock()

    def show_status_message(self, msg, duration=0):
        pass

    # Unused by this plugin but part of the standard API
    def add_menu_action(self, path, callback, **kwargs): pass
    def register_save_handler(self, fn): pass
    def register_load_handler(self, fn): pass
    def register_document_reset_handler(self, fn): pass
    def register_file_opener(self, ext, fn, priority=0): pass
    def register_drop_handler(self, fn, priority=0): pass
    def add_analysis_tool(self, label, fn): pass
    def add_toolbar_action(self, fn, text, icon=None, tooltip=None): pass


# ---------------------------------------------------------------------------
# Tests: metadata
# ---------------------------------------------------------------------------

class TestMetadata(unittest.TestCase):
    def test_plugin_name(self):
        self.assertEqual(PLUGIN_NAME, "Rotation Giffer")

    def test_plugin_version_is_semver(self):
        parts = PLUGIN_VERSION.split(".")
        self.assertEqual(len(parts), 3)
        for p in parts:
            self.assertTrue(p.isdigit(), f"Non-numeric version part: {p!r}")


# ---------------------------------------------------------------------------
# Tests: initialize registers export action
# ---------------------------------------------------------------------------

class TestInitialize(unittest.TestCase):
    def setUp(self):
        self.ctx = _StubContext()
        initialize(self.ctx)

    def test_registers_one_export_action(self):
        self.assertEqual(len(self.ctx._export_actions), 1)

    def test_export_action_label_mentions_gif(self):
        label, _ = self.ctx._export_actions[0]
        self.assertIn("GIF", label)

    def test_export_action_is_callable(self):
        _, callback = self.ctx._export_actions[0]
        self.assertTrue(callable(callback))

    def test_second_initialize_also_registers(self):
        ctx2 = _StubContext()
        initialize(ctx2)
        self.assertEqual(len(ctx2._export_actions), 1)


# ---------------------------------------------------------------------------
# Tests: show_giffer_dialog singleton
# ---------------------------------------------------------------------------

class TestShowGifferDialog(unittest.TestCase):
    def test_reuses_existing_window(self):
        """If a window already exists, show_giffer_dialog should reuse it."""
        ctx = _StubContext()
        fake_win = MagicMock()
        ctx._windows["rotation_giffer_dialog"] = fake_win
        _pkg.show_giffer_dialog(ctx)
        fake_win.show.assert_called_once()
        fake_win.raise_.assert_called_once()


# ---------------------------------------------------------------------------
# Real PluginContext tier (local dev + CI with cloned main app)
# ---------------------------------------------------------------------------

_MAIN_APP_CANDIDATES = [
    os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..",
                     "python_molecular_editor", "moleditpy", "src")
    ),
    os.environ.get("CI_MAIN_APP_SRC", ""),
]
_MAIN_APP_SRC = next(
    (p for p in _MAIN_APP_CANDIDATES if p and os.path.isdir(p)),
    None,
)
HAS_MAIN_APP = _MAIN_APP_SRC is not None

try:
    import pytest
    _skipif = pytest.mark.skipif(
        not HAS_MAIN_APP,
        reason="main app not found; clone python_molecular_editor or set CI_MAIN_APP_SRC",
    )
except ImportError:
    def _skipif(cls):
        return unittest.skip("pytest not available")(cls)


def _clear_qt_stubs():
    """Remove fake PyQt6 stub modules so real PyQt6 can be imported by moleditpy."""
    to_remove = [
        k for k in list(sys.modules)
        if k.startswith("PyQt6") and not hasattr(sys.modules[k], "__file__")
    ]
    for k in to_remove:
        del sys.modules[k]
    # Clear any moleditpy import that may have been attempted with stubs
    for k in [k for k in list(sys.modules) if k.startswith("moleditpy")]:
        del sys.modules[k]


@_skipif
class TestWithRealPluginContext(unittest.TestCase):
    """Verify initialize() works with the actual MoleditPy PluginContext."""

    @classmethod
    def setUpClass(cls):
        if not HAS_MAIN_APP:
            return
        # Load plugin_interface.py directly to avoid triggering moleditpy/__init__.py
        # which imports PyQt6 and conflicts with PySide6 loaded by pytest-qt on Windows.
        import importlib.util as _ilu
        _pi_path = os.path.join(_MAIN_APP_SRC, 'moleditpy', 'plugins', 'plugin_interface.py')
        _spec = _ilu.spec_from_file_location('moleditpy.plugins.plugin_interface', _pi_path)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        cls.PluginContext = _mod.PluginContext
        mock_manager = MagicMock()
        mock_manager.get_main_window.return_value = MagicMock()
        cls.real_ctx = cls.PluginContext(mock_manager, PLUGIN_NAME)

    def test_real_initialize_does_not_raise(self):
        try:
            initialize(self.real_ctx)
        except Exception as e:
            self.fail(f"initialize(real_context) raised: {e}")

    def test_real_context_is_plugincontext_instance(self):
        self.assertIsInstance(self.real_ctx, self.PluginContext)

    def test_stub_interface_matches_real(self):
        for method in ["add_export_action", "get_main_window"]:
            self.assertTrue(
                hasattr(self.PluginContext, method),
                f"Real PluginContext missing: {method}",
            )


if __name__ == "__main__":
    unittest.main()
