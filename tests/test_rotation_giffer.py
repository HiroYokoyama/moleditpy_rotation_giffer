"""
tests/test_rotation_giffer.py
Unit tests for the rotation giffer plugin.
"""

# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,protected-access,invalid-name

import importlib.util as importlib_util
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np


# ---------------------------------------------------------------------------
# Install Qt stubs so that Pyvista / PyQt6 can be imported / mocked cleanly
# in a headless test environment.
# ---------------------------------------------------------------------------
def _install_qt_stubs():
    # PyQt6.QtCore stubs
    qt_core = types.ModuleType("PyQt6.QtCore")
    class _Qt:
        class AlignmentFlag:
            AlignRight = None
        class Orientation:
            Horizontal = None
        class CursorShape:
            PointingHandCursor = None
    qt_core.Qt = _Qt

    class _QCoreApplication:
        @staticmethod
        def processEvents():
            pass
    qt_core.QCoreApplication = _QCoreApplication

    pyqt6 = types.ModuleType("PyQt6")
    pyqt6.QtCore = qt_core
    sys.modules["PyQt6"] = pyqt6
    sys.modules["PyQt6.QtCore"] = qt_core

    # PyQt6.QtWidgets stubs
    qt_widgets = types.ModuleType("PyQt6.QtWidgets")
    class _QDialog:
        def __init__(self, parent=None):
            self._parent = parent
            self._window_title = ""
            self._accepted = False
        def setWindowTitle(self, title):
            self._window_title = title
        def accept(self):
            self._accepted = True
        def reject(self):
            self._accepted = False
        def show(self):
            pass
    qt_widgets.QDialog = _QDialog

    for name in [
        "QVBoxLayout", "QHBoxLayout", "QLabel", "QComboBox", "QSpinBox",
        "QCheckBox", "QPushButton", "QFormLayout"
    ]:
        setattr(qt_widgets, name, lambda *args, **kwargs: MagicMock())

    qt_widgets.QFileDialog = MagicMock()
    qt_widgets.QMessageBox = MagicMock()

    pyqt6.QtWidgets = qt_widgets
    sys.modules["PyQt6.QtWidgets"] = qt_widgets


_install_qt_stubs()


# Now import/load rotation_giffer module
def _load_module_direct(relpath, module_name):
    src = os.path.join(os.path.dirname(__file__), "..", relpath)
    src = os.path.normpath(src)
    spec = importlib_util.spec_from_file_location(module_name, src)
    mod = importlib_util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


giffer_mod = _load_module_direct("rotation_giffer.py", "rotation_giffer_under_test")


class TestGifferMetadata(unittest.TestCase):
    """Verify plugin metadata."""
    def test_metadata(self):
        self.assertEqual(giffer_mod.PLUGIN_NAME, "Rotation Giffer")
        self.assertEqual(giffer_mod.PLUGIN_AUTHOR, "HiroYokoyama")
        self.assertEqual(giffer_mod.PLUGIN_CATEGORY, "Export")


class TestGifferInitialize(unittest.TestCase):
    """Verify plugin initialization."""
    def test_initialize(self):
        context = MagicMock()
        giffer_mod.initialize(context)
        context.add_export_action.assert_called_once()
        args, _ = context.add_export_action.call_args
        self.assertEqual(args[0], "Generate Rotation GIF...")


class TestShowGifferDialog(unittest.TestCase):
    """Verify show_giffer_dialog singleton behavior."""
    def test_show_giffer_dialog_active(self):
        context = MagicMock()
        active_window = MagicMock()
        context.get_window.return_value = active_window

        giffer_mod.show_giffer_dialog(context)

        context.get_window.assert_called_once_with("rotation_giffer_dialog")
        active_window.show.assert_called_once()
        active_window.raise_.assert_called_once()

    @patch("rotation_giffer_under_test.GifferDialog")
    def test_show_giffer_dialog_new(self, mock_dialog_class):
        context = MagicMock()
        context.get_window.return_value = None
        mock_mw = MagicMock()
        context.get_main_window.return_value = mock_mw

        giffer_mod.show_giffer_dialog(context)

        context.get_window.assert_called_once_with("rotation_giffer_dialog")
        context.get_main_window.assert_called_once()
        mock_dialog_class.assert_called_once_with(context, mock_mw)
        dialog_instance = mock_dialog_class.return_value
        context.register_window.assert_called_once_with(
            "rotation_giffer_dialog", dialog_instance
        )
        dialog_instance.show.assert_called_once()


class TestOrbitCameraMath(unittest.TestCase):
    """Verify the vector rotation mathematics of _orbit_camera."""
    def setUp(self):
        self.plotter = MagicMock()
        self.camera = MagicMock()
        self.plotter.camera = self.camera
        self.renderer = MagicMock()
        self.plotter.renderer = self.renderer

        self.dialog = giffer_mod.GifferDialog.__new__(giffer_mod.GifferDialog)

    def test_orbit_z_axis_global(self):
        # Camera looking along Y axis at origin, Up along Z
        # fp = (0, 0, 0), pos = (0, -10, 0), up = (0, 0, 1)
        self.camera.focal_point = (0.0, 0.0, 0.0)
        self.camera.position = (0.0, -10.0, 0.0)
        self.camera.up = (0.0, 0.0, 1.0)

        # Orbit 90 degrees around Z axis (axis_idx=0)
        # step = 90, frame_idx = 1
        # theta = -step = -90 degrees
        self.dialog._orbit_camera(
            self.plotter, axis_idx=0, step=90, frame_idx=1,
            initial_pos=self.camera.position, initial_up=self.camera.up
        )

        # Call verification
        self.camera.SetPosition.assert_called_once()
        self.camera.SetViewUp.assert_called_once()

        pos_args = self.camera.SetPosition.call_args[0]
        up_args = self.camera.SetViewUp.call_args[0]

        # Expected position rotated around Z axis by -90 degrees (clockwise):
        # (0, -10, 0) rotated by -90 deg:
        # x_new = x*cos(-90) - y*sin(-90) = 0 - (-10)*(-1) = -10
        # y_new = x*sin(-90) + y*cos(-90) = 0 + (-10)*(0) = 0
        # Wait, Rodrigues formula rotates around axis. Let's check:
        # view_vec = pos - fp = (0, -10, 0)
        # k = (0, 0, 1)
        # theta = -90 deg = -pi/2
        # v_rot = v * cos(theta) + (k x v) * sin(theta)
        # k x v = (0, 0, 1) x (0, -10, 0) = (10, 0, 0)
        # v_rot = (0, -10, 0) * 0 + (10, 0, 0) * (-1) = (-10, 0, 0)
        expected_pos = np.array([-10.0, 0.0, 0.0])
        expected_up = np.array([0.0, 0.0, 1.0])

        self.assertTrue(np.allclose(pos_args, expected_pos))
        self.assertTrue(np.allclose(up_args, expected_up))
        self.renderer.ResetCameraClippingRange.assert_called_once()
        self.plotter.render.assert_called_once()


class TestGifferDialogExecution(unittest.TestCase):
    """Verify dialog validation, settings retrieval, and GIF output loops."""
    def setUp(self):
        self.context = MagicMock()
        self.dialog = giffer_mod.GifferDialog(self.context)
        self.dialog.axis_combo = MagicMock()
        self.dialog.angle_spin = MagicMock()
        self.dialog.frames_spin = MagicMock()
        self.dialog.fps_spin = MagicMock()
        self.dialog.transparency_check = MagicMock()
        self.dialog.hq_check = MagicMock()
        self.dialog.inverse_check = MagicMock()
        self.dialog.accept = MagicMock()

    def test_generate_gif_no_molecule(self):
        self.context.current_molecule = None
        with patch("PyQt6.QtWidgets.QMessageBox.warning") as mock_warning:
            self.dialog.generate_gif()
            mock_warning.assert_called_once()
            self.context.plotter.assert_not_called()

    @patch("PyQt6.QtWidgets.QFileDialog.getSaveFileName")
    def test_generate_gif_cancel_save(self, mock_get_save_filename):
        self.context.current_molecule = MagicMock()
        mock_get_save_filename.return_value = ("", "")

        self.dialog.generate_gif()
        self.context.plotter.open_gif.assert_not_called()

    @patch("PyQt6.QtWidgets.QFileDialog.getSaveFileName")
    def test_generate_gif_standard_loop(self, mock_get_save_filename):
        self.context.current_molecule = MagicMock()
        mock_get_save_filename.return_value = ("test_out.gif", "GIF Files (*.gif)")

        # Mock MainWindow with loaded file path
        mw = MagicMock()
        mw.init_manager.current_file_path = os.path.normpath("/path/to/my_molecule.xyz")
        self.context.get_main_window.return_value = mw

        # Configure settings mock
        self.dialog.axis_combo.currentIndex.return_value = 0 # Z axis
        self.dialog.angle_spin.value.return_value = 360
        self.dialog.frames_spin.value.return_value = 10
        self.dialog.fps_spin.value.return_value = 10
        self.dialog.transparency_check.isChecked.return_value = False
        self.dialog.hq_check.isChecked.return_value = False
        self.dialog.inverse_check.isChecked.return_value = False

        # Set up plotter mock
        plotter = MagicMock()
        self.context.plotter = plotter
        plotter.camera_position = ((0, 0, 10), (0, 0, 0), (0, 1, 0))
        plotter.camera.position = (0, 0, 10)
        plotter.camera.up = (0, 1, 0)
        plotter.camera.focal_point = (0, 0, 0)
        plotter.mwriter = MagicMock()

        self.dialog.generate_gif()

        # Check interaction and verify default save path matches base name + _ZRot suffix
        expected_default_path = os.path.normpath("/path/to/my_molecule_ZRot.gif")
        mock_get_save_filename.assert_called_once()
        args, _ = mock_get_save_filename.call_args
        self.assertEqual(args[2], expected_default_path)

        plotter.open_gif.assert_called_once_with("test_out.gif")
        self.assertEqual(plotter.write_frame.call_count, 10)
        plotter.mwriter.close.assert_called_once()
        self.dialog.accept.assert_called_once()

    @patch("PyQt6.QtWidgets.QFileDialog.getSaveFileName")
    def test_generate_gif_restores_without_full_redraw(self, mock_get_save_filename):
        # Restoring the camera must NOT trigger a full draw_molecule_3d() rebuild
        # (context.refresh_3d_view); a plain plotter.render() is enough.
        self.context.current_molecule = MagicMock()
        mock_get_save_filename.return_value = ("test_out.gif", "GIF Files (*.gif)")

        mw = MagicMock()
        mw.init_manager.current_file_path = os.path.normpath("/path/to/my_molecule.xyz")
        self.context.get_main_window.return_value = mw

        self.dialog.axis_combo.currentIndex.return_value = 0
        self.dialog.angle_spin.value.return_value = 360
        self.dialog.frames_spin.value.return_value = 5
        self.dialog.fps_spin.value.return_value = 10
        self.dialog.transparency_check.isChecked.return_value = False
        self.dialog.hq_check.isChecked.return_value = False
        self.dialog.inverse_check.isChecked.return_value = False

        plotter = MagicMock()
        self.context.plotter = plotter
        plotter.camera_position = ((0, 0, 10), (0, 0, 0), (0, 1, 0))
        plotter.camera.position = (0, 0, 10)
        plotter.camera.up = (0, 1, 0)
        plotter.camera.focal_point = (0, 0, 0)
        plotter.mwriter = MagicMock()

        self.dialog.generate_gif()

        # The expensive full-molecule redraw must never be invoked on restore.
        self.context.refresh_3d_view.assert_not_called()
        # Camera state is restored and the existing scene is re-rendered instead.
        plotter.renderer.ResetCameraClippingRange.assert_called()
        plotter.render.assert_called()

    @patch("PyQt6.QtWidgets.QFileDialog.getSaveFileName")
    def test_generate_gif_inverse_rotation(self, mock_get_save_filename):
        self.context.current_molecule = MagicMock()
        mock_get_save_filename.return_value = ("test_inverse.gif", "GIF Files (*.gif)")

        # Mock MainWindow with loaded file path
        mw = MagicMock()
        mw.init_manager.current_file_path = os.path.normpath("/path/to/my_molecule.xyz")
        self.context.get_main_window.return_value = mw

        # Configure settings mock: inverse rotation checked, and axis_idx = 1 (X Axis)
        self.dialog.axis_combo.currentIndex.return_value = 1
        self.dialog.angle_spin.value.return_value = 360
        self.dialog.frames_spin.value.return_value = 4
        self.dialog.fps_spin.value.return_value = 4
        self.dialog.transparency_check.isChecked.return_value = False
        self.dialog.hq_check.isChecked.return_value = False
        self.dialog.inverse_check.isChecked.return_value = True

        # Set up plotter mock
        plotter = MagicMock()
        self.context.plotter = plotter
        plotter.camera_position = ((0, -10, 0), (0, 0, 0), (0, 0, 1))
        plotter.camera.position = (0, -10, 0)
        plotter.camera.up = (0, 0, 1)
        plotter.camera.focal_point = (0, 0, 0)
        plotter.mwriter = MagicMock()

        # Track _orbit_camera calls to see the step direction
        with patch.object(self.dialog, "_orbit_camera") as mock_orbit:
            self.dialog.generate_gif()

            # Check default path matches base name + _XRot suffix
            expected_default_path = os.path.normpath("/path/to/my_molecule_XRot.gif")
            mock_get_save_filename.assert_called_once()
            args, _ = mock_get_save_filename.call_args
            self.assertEqual(args[2], expected_default_path)

            # 360 deg / 4 frames = 90 deg.
            # inverse is True => direction_multiplier = -1 => step = -90.0
            # Calls to _orbit_camera should use step = -90.0, axis_idx = 1 (X Axis)
            mock_orbit.assert_has_calls([
                call(plotter, 1, -90.0, 1, (0, -10, 0), (0, 0, 1)),
                call(plotter, 1, -90.0, 2, (0, -10, 0), (0, 0, 1)),
                call(plotter, 1, -90.0, 3, (0, -10, 0), (0, 0, 1)),
                call(plotter, 1, -90.0, 4, (0, -10, 0), (0, 0, 1)),
            ])


if __name__ == "__main__":
    unittest.main()
