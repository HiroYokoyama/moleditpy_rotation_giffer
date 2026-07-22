"""
tests/test_rotation_giffer_coverage.py
Additional coverage tests for rotation_giffer.py:
- HAS_PIL import fallback
- setup_ui() branch when PIL is unavailable
- default save-path fallback when no current file is loaded
- transparent/HQ Pillow GIF-assembly path (frame capture + palette quantization)
- anti-aliasing enable/disable error branches
- generate_gif() top-level exception branch
- _orbit_camera() axis branches (X/Y global, roll, elevation, azimuth, unknown)
  and the degenerate-vector early returns
"""

# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,protected-access,invalid-name

import importlib.util as importlib_util
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Reuse the exact same Qt-stub installation (and already-loaded plugin module)
# from test_rotation_giffer.py. Re-installing the stubs or reloading the
# module under a second name would leave rotation_giffer.py bound to a
# different QFileDialog/QMessageBox stub object than the one @patch()
# targets, silently breaking unpacking of getSaveFileName()'s return value.
import test_rotation_giffer as _base_test_module
from test_rotation_giffer import _load_module_direct

giffer_mod = _base_test_module.giffer_mod


class TestHasPilImportFallback(unittest.TestCase):
    """Verify HAS_PIL is set to False when the PIL import fails (lines 31-32)."""

    def test_has_pil_false_when_pil_missing(self):
        class _BlockPILFinder:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "PIL" or fullname.startswith("PIL."):
                    raise ImportError(f"blocked: {fullname}")
                return None

        blocker = _BlockPILFinder()

        # `patch.dict(sys.modules)` snapshots the *entire* dict on entry and,
        # on exit, unconditionally restores it byte-for-byte (clears + re-
        # populates from the saved copy) -- regardless of what the blocked
        # import attempt does to sys.modules in between. This is stronger
        # than manually deleting/re-adding the PIL* keys we think are
        # relevant: it also undoes any partial/stub entries the blocked
        # import machinery might otherwise leave behind, which is what was
        # corrupting real Pillow (`PIL.Image` missing `fromarray`) for the
        # tests that ran after this one on a fresh CI runner.
        with patch.dict(sys.modules):
            for name in [n for n in sys.modules if n == "PIL" or n.startswith("PIL.")]:
                del sys.modules[name]

            sys.meta_path.insert(0, blocker)
            try:
                mod = _load_module_direct(
                    "rotation_giffer.py", "rotation_giffer_under_test_nopil"
                )
                self.assertFalse(mod.HAS_PIL)
            finally:
                sys.meta_path.remove(blocker)
                sys.modules.pop("rotation_giffer_under_test_nopil", None)

        # Sanity-check that the restore left genuine, working Pillow behind
        # for every subsequent test in this process (guards against any
        # regression in the isolation strategy above).
        from PIL import Image as _RealImage  # pylint: disable=import-outside-toplevel
        self.assertTrue(hasattr(_RealImage, "fromarray"))


class TestSetupUiWithoutPil(unittest.TestCase):
    """Verify setup_ui() disables transparency/HQ options when PIL is absent
    (lines 131-134)."""

    def test_setup_ui_disables_transparency_and_hq_without_pil(self):
        with patch.object(giffer_mod, "HAS_PIL", False):
            context = MagicMock()
            dialog = giffer_mod.GifferDialog(context)

            dialog.transparency_check.setEnabled.assert_any_call(False)
            dialog.transparency_check.setChecked.assert_any_call(False)
            dialog.hq_check.setEnabled.assert_any_call(False)
            dialog.transparency_check.setToolTip.assert_called_once()
            tooltip_args = dialog.transparency_check.setToolTip.call_args[0]
            self.assertIn("Pillow", tooltip_args[0])


class TestGenerateGifDefaultPathFallback(unittest.TestCase):
    """Verify the default save path falls back to 'molecule<suffix>.gif' when
    no current file path is available (line 193)."""

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

    @patch("PyQt6.QtWidgets.QFileDialog.getSaveFileName")
    def test_default_path_fallback_no_main_window_file(self, mock_get_save):
        self.context.current_molecule = MagicMock()
        # No main window / no init_manager.current_file_path -> falls through.
        mw = MagicMock()
        mw.init_manager.current_file_path = None
        self.context.get_main_window.return_value = mw

        self.dialog.axis_combo.currentIndex.return_value = 0  # -> "_ZRot"
        mock_get_save.return_value = ("", "")  # cancel immediately after path calc

        self.dialog.generate_gif()

        mock_get_save.assert_called_once()
        args, _ = mock_get_save.call_args
        self.assertEqual(args[2], "molecule_ZRot.gif")

    @patch("PyQt6.QtWidgets.QFileDialog.getSaveFileName")
    def test_default_path_fallback_no_main_window(self, mock_get_save):
        self.context.current_molecule = MagicMock()
        self.context.get_main_window.return_value = None

        self.dialog.axis_combo.currentIndex.return_value = 5  # -> "_AzimRot"
        mock_get_save.return_value = ("", "")

        self.dialog.generate_gif()

        args, _ = mock_get_save.call_args
        self.assertEqual(args[2], "molecule_AzimRot.gif")


def _make_rgba_frame(alpha_value):
    """Build a tiny 4x4 RGBA numpy frame with a uniform alpha channel."""
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    arr[:, :, 0] = 200  # R
    arr[:, :, 1] = 100  # G
    arr[:, :, 2] = 50   # B
    arr[:, :, 3] = alpha_value  # A
    return arr


class TestGenerateGifTransparentPillowPath(unittest.TestCase):
    """Exercise the real Pillow transparent-GIF assembly path (use_transparency
    and HAS_PIL are both True) -- lines 236-257, plus the anti-aliasing
    enable/disable branches around it."""

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

        self.context.current_molecule = MagicMock()
        mw = MagicMock()
        mw.init_manager.current_file_path = None
        self.context.get_main_window.return_value = mw

        self.dialog.axis_combo.currentIndex.return_value = 0
        self.dialog.angle_spin.value.return_value = 360
        self.dialog.frames_spin.value.return_value = 3
        self.dialog.fps_spin.value.return_value = 10
        self.dialog.inverse_check.isChecked.return_value = False

        self.plotter = MagicMock()
        self.context.plotter = self.plotter
        self.plotter.camera_position = ((0, 0, 10), (0, 0, 0), (0, 1, 0))
        self.plotter.camera.position = (0, 0, 10)
        self.plotter.camera.up = (0, 1, 0)
        self.plotter.camera.focal_point = (0, 0, 0)

        # Alternate alpha per call so the HQ mask-paste path has something
        # interesting to threshold.
        self._frame_alphas = [255, 10, 200]
        self._call_idx = 0

        def _screenshot(*args, **kwargs):  # pylint: disable=unused-argument
            alpha = self._frame_alphas[self._call_idx % len(self._frame_alphas)]
            self._call_idx += 1
            return _make_rgba_frame(alpha)

        self.plotter.screenshot.side_effect = _screenshot

        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.out_path = os.path.join(self.tmpdir.name, "out.gif")

    @patch("PyQt6.QtWidgets.QFileDialog.getSaveFileName")
    def test_transparent_hq_pillow_path_writes_real_gif(self, mock_get_save):
        mock_get_save.return_value = (self.out_path, "GIF Files (*.gif)")
        self.dialog.transparency_check.isChecked.return_value = True
        self.dialog.hq_check.isChecked.return_value = True

        self.dialog.generate_gif()

        self.plotter.enable_anti_aliasing.assert_called_once_with('ssaa')
        self.assertEqual(self.plotter.screenshot.call_count, 3)
        self.assertTrue(os.path.exists(self.out_path))
        self.assertGreater(os.path.getsize(self.out_path), 0)
        self.dialog.accept.assert_called_once()
        self.plotter.render.assert_called()

    @patch("PyQt6.QtWidgets.QFileDialog.getSaveFileName")
    def test_transparent_no_hq_pillow_path_writes_real_gif(self, mock_get_save):
        mock_get_save.return_value = (self.out_path, "GIF Files (*.gif)")
        self.dialog.transparency_check.isChecked.return_value = True
        self.dialog.hq_check.isChecked.return_value = False

        self.dialog.generate_gif()

        self.plotter.enable_anti_aliasing.assert_not_called()
        self.assertEqual(self.plotter.screenshot.call_count, 3)
        self.assertTrue(os.path.exists(self.out_path))
        self.dialog.accept.assert_called_once()

    @patch("PyQt6.QtWidgets.QFileDialog.getSaveFileName")
    def test_anti_aliasing_enable_failure_is_logged_and_swallowed(self, mock_get_save):
        mock_get_save.return_value = (self.out_path, "GIF Files (*.gif)")
        self.dialog.transparency_check.isChecked.return_value = True
        self.dialog.hq_check.isChecked.return_value = True
        self.plotter.enable_anti_aliasing.side_effect = RuntimeError("no ssaa support")

        # Should not raise -- the failure is caught and logged, generation continues.
        self.dialog.generate_gif()

        self.plotter.enable_anti_aliasing.assert_called_once_with('ssaa')
        self.assertTrue(os.path.exists(self.out_path))
        self.dialog.accept.assert_called_once()

    @patch("PyQt6.QtWidgets.QFileDialog.getSaveFileName")
    def test_disable_anti_aliasing_failure_is_logged_and_swallowed(self, mock_get_save):
        mock_get_save.return_value = (self.out_path, "GIF Files (*.gif)")
        self.dialog.transparency_check.isChecked.return_value = True
        self.dialog.hq_check.isChecked.return_value = True
        # render_window must be truthy for disable_anti_aliasing() to be attempted.
        self.plotter.render_window = MagicMock()
        self.plotter.disable_anti_aliasing.side_effect = RuntimeError("cannot disable")

        # Should not raise -- caught/logged in the finally block.
        self.dialog.generate_gif()

        self.plotter.disable_anti_aliasing.assert_called_once()
        self.dialog.accept.assert_called_once()
        # Camera restore still happens even though disabling AA failed.
        self.plotter.render.assert_called()


class TestGenerateGifExceptionBranch(unittest.TestCase):
    """Verify the top-level try/except reports errors via QMessageBox.critical
    (lines 279-281) without raising, and still restores camera state."""

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

        self.context.current_molecule = MagicMock()
        mw = MagicMock()
        mw.init_manager.current_file_path = None
        self.context.get_main_window.return_value = mw

        self.dialog.axis_combo.currentIndex.return_value = 0
        self.dialog.angle_spin.value.return_value = 360
        self.dialog.frames_spin.value.return_value = 4
        self.dialog.fps_spin.value.return_value = 10
        self.dialog.inverse_check.isChecked.return_value = False
        self.dialog.transparency_check.isChecked.return_value = False
        self.dialog.hq_check.isChecked.return_value = False

        self.plotter = MagicMock()
        self.context.plotter = self.plotter
        self.plotter.camera_position = ((0, 0, 10), (0, 0, 0), (0, 1, 0))
        self.plotter.camera.position = (0, 0, 10)
        self.plotter.camera.up = (0, 1, 0)
        self.plotter.camera.focal_point = (0, 0, 0)

    @patch("PyQt6.QtWidgets.QFileDialog.getSaveFileName")
    def test_open_gif_failure_shows_critical_and_restores_camera(self, mock_get_save):
        mock_get_save.return_value = ("broken.gif", "GIF Files (*.gif)")
        self.plotter.open_gif.side_effect = RuntimeError("disk full")

        with patch("PyQt6.QtWidgets.QMessageBox.critical") as mock_critical:
            self.dialog.generate_gif()
            mock_critical.assert_called_once()

        # accept() must NOT be called on failure.
        self.dialog.accept.assert_not_called()
        # Camera restoration still happens in `finally`.
        self.assertEqual(self.plotter.camera_position, ((0, 0, 10), (0, 0, 0), (0, 1, 0)))
        self.plotter.render.assert_called()


class TestOrbitCameraAxisBranches(unittest.TestCase):
    """Cover the remaining axis-selection branches and degenerate-vector
    early returns in _orbit_camera (lines 329, 339-350, 355)."""

    def setUp(self):
        self.plotter = MagicMock()
        self.camera = MagicMock()
        self.plotter.camera = self.camera
        self.renderer = MagicMock()
        self.plotter.renderer = self.renderer
        self.dialog = giffer_mod.GifferDialog.__new__(giffer_mod.GifferDialog)

    def _run(self, axis_idx, pos=(0.0, -10.0, 0.0), up=(0.0, 0.0, 1.0), fp=(0.0, 0.0, 0.0)):
        self.camera.focal_point = fp
        self.camera.position = pos
        self.camera.up = up
        self.dialog._orbit_camera(
            self.plotter, axis_idx=axis_idx, step=30, frame_idx=1,
            initial_pos=pos, initial_up=up,
        )

    def test_x_axis_global(self):
        self._run(axis_idx=1)
        self.camera.SetPosition.assert_called_once()
        self.camera.SetViewUp.assert_called_once()

    def test_y_axis_global(self):
        self._run(axis_idx=2)
        self.camera.SetPosition.assert_called_once()
        self.camera.SetViewUp.assert_called_once()

    def test_roll_axis(self):
        self._run(axis_idx=3)
        self.camera.SetPosition.assert_called_once()
        self.camera.SetViewUp.assert_called_once()

    def test_elevation_axis(self):
        self._run(axis_idx=4)
        self.camera.SetPosition.assert_called_once()
        self.camera.SetViewUp.assert_called_once()

    def test_azimuth_axis(self):
        self._run(axis_idx=5)
        self.camera.SetPosition.assert_called_once()
        self.camera.SetViewUp.assert_called_once()

    def test_unknown_axis_returns_without_action(self):
        self._run(axis_idx=99)
        self.camera.SetPosition.assert_not_called()
        self.camera.SetViewUp.assert_not_called()
        self.renderer.ResetCameraClippingRange.assert_not_called()

    def test_degenerate_view_vector_returns_early(self):
        # pos == fp -> view_vec is zero -> view_norm < eps -> early return (line 329).
        self._run(axis_idx=0, pos=(0.0, 0.0, 0.0), fp=(0.0, 0.0, 0.0))
        self.camera.SetPosition.assert_not_called()
        self.camera.SetViewUp.assert_not_called()

    def test_degenerate_up_vector_returns_early(self):
        # up is zero -> up_norm < eps -> early return (line 329).
        self._run(axis_idx=0, up=(0.0, 0.0, 0.0))
        self.camera.SetPosition.assert_not_called()
        self.camera.SetViewUp.assert_not_called()

    def test_degenerate_rotation_axis_returns_early(self):
        # Elevation axis = cross(up_dir, view_dir). Make up parallel to view
        # so right_dir ~ 0 -> axis_norm < eps -> early return (line 355).
        self._run(axis_idx=4, pos=(0.0, 0.0, 10.0), up=(0.0, 0.0, 1.0), fp=(0.0, 0.0, 0.0))
        self.camera.SetPosition.assert_not_called()
        self.camera.SetViewUp.assert_not_called()


if __name__ == "__main__":
    unittest.main()
