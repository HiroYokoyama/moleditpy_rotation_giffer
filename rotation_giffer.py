"""
Rotation Giffer Plugin for MoleditPy.
Allows generating rotating GIF animations around global or view axes by orbiting the camera.
"""

# pylint: disable=too-many-instance-attributes,too-many-locals,too-many-branches
# pylint: disable=too-many-statements,too-many-arguments,too-many-positional-arguments
# pylint: disable=no-name-in-module

import math
import logging
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                             QComboBox, QSpinBox, QCheckBox, QPushButton,
                             QFileDialog, QMessageBox, QFormLayout)
import numpy as np

# Set up logging for the plugin
logger = logging.getLogger(__name__)

# Attempt to load PIL for transparent GIF generation and color quantization
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# --- Plugin Metadata ---
PLUGIN_NAME = "Rotation Giffer"
PLUGIN_VERSION = "1.2.0"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = "Creates a rotating GIF around global or view axes by orbiting the camera."
PLUGIN_CATEGORY = "Export"


def initialize(context):
    """
    Register the tool in the Export menu.
    """
    context.add_export_action("Generate Rotation GIF...", lambda: show_giffer_dialog(context))


def show_giffer_dialog(context):
    """
    Singleton Pattern: Check if the window is already active, otherwise create it.
    """
    win = context.get_window("rotation_giffer_dialog")
    if win:
        win.show()
        win.raise_()
        return

    mw = context.get_main_window()
    dialog = GifferDialog(context, mw)
    # Safely register the window to prevent garbage collection
    context.register_window("rotation_giffer_dialog", dialog)
    dialog.show()


class GifferDialog(QDialog):
    """
    Dialog window for specifying GIF generation settings and executing the generation.
    """
    def __init__(self, context, parent=None):
        super().__init__(parent)
        self.context = context
        self.setWindowTitle("Rotation Giffer")
        self.setup_ui()

    def setup_ui(self):
        """
        Setup the layout and input widgets for the dialog.
        """
        layout = QVBoxLayout(self)
        form = QFormLayout()

        # Axis Selection (Absolute Global Axes + Local View Axes)
        self.axis_combo = QComboBox()
        self.axis_combo.addItems([
            "Z Axis (Global)",
            "X Axis (Global)",
            "Y Axis (Global)",
            "Roll (Spin around line of sight)",
            "Elevation (Pitch up/down)",
            "Azimuth (Yaw left/right)"
        ])
        form.addRow("Rotation Axis:", self.axis_combo)

        # --- Inverse Rotation Checkbox ---
        self.inverse_check = QCheckBox("Inverse Rotation (Reverse Direction)")
        self.inverse_check.setChecked(False)
        form.addRow("Direction:", self.inverse_check)

        # Total Rotation Angle
        self.angle_spin = QSpinBox()
        self.angle_spin.setRange(10, 3600)
        self.angle_spin.setValue(360)
        self.angle_spin.setSuffix(" °")
        form.addRow("Total Rotation:", self.angle_spin)

        # Frames (Controls smoothness)
        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(5, 360)
        self.frames_spin.setValue(36)
        form.addRow("Total Frames:", self.frames_spin)

        # Speed Control (FPS)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(5)
        self.fps_spin.setSuffix(" FPS")
        form.addRow("Speed:", self.fps_spin)

        # Transparent Background Checkbox
        self.transparency_check = QCheckBox("Transparent Background")
        self.transparency_check.setChecked(True)

        # High Quality Colors & Rendering Checkbox
        self.hq_check = QCheckBox("High Quality Colors & Anti-Aliasing")
        self.hq_check.setChecked(True)

        # Disable advanced options if PIL is missing in the environment
        if not HAS_PIL:
            self.transparency_check.setEnabled(False)
            self.transparency_check.setChecked(False)
            self.hq_check.setEnabled(False)
            self.transparency_check.setToolTip(
                "PIL (Pillow) library is required to export transparent GIFs."
            )

        form.addRow("Options:", self.transparency_check)
        form.addRow("", self.hq_check)

        layout.addLayout(form)

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.btn_generate = QPushButton("Generate GIF")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_generate.clicked.connect(self.generate_gif)
        self.btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(self.btn_generate)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

    def generate_gif(self):
        """
        Gathers settings, performs the orbital camera math frame-by-frame,
        captures screenshots, saves the GIF file, and restores the original camera.
        """
        # Validate that a molecule is present
        if not self.context.current_molecule:
            QMessageBox.warning(self, "Warning", "No molecule loaded.")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Rotating GIF", "", "GIF Files (*.gif)"
        )
        if not save_path:
            return

        # Retrieve user inputs
        axis_idx = self.axis_combo.currentIndex()
        total_angle = self.angle_spin.value()
        frames = self.frames_spin.value()
        fps = self.fps_spin.value()
        use_transparency = self.transparency_check.isChecked()
        use_hq = self.hq_check.isChecked()

        # --- Retrieve inverse selection ---
        use_inverse = self.inverse_check.isChecked()

        # Calculate duration per frame in milliseconds
        frame_duration = int(1000 / fps)

        # Direct access to the PyVista plotter
        plotter = self.context.plotter

        # ---  Apply the inversion to the step calculation ---
        direction_multiplier = -1 if use_inverse else 1
        step = (total_angle / frames) * direction_multiplier

        # Save exact initial camera state so it can be restored perfectly later
        initial_cpos = plotter.camera_position        # Full 3x3 state for restoring later
        initial_pos_coord = plotter.camera.position   # Just the (X, Y, Z) for the math
        initial_cam_up = plotter.camera.up

        try:
            self.context.show_status_message("Generating GIF... Please wait.", 10000)

            if use_hq:
                try:
                    plotter.enable_anti_aliasing('ssaa')
                except Exception as aa_err:  # pylint: disable=broad-exception-caught
                    logger.warning("Failed to enable anti-aliasing: %s", aa_err)

            if use_transparency and HAS_PIL:
                images = []
                for i in range(frames):
                    img_array = plotter.screenshot(transparent_background=True, return_img=True)
                    img = Image.fromarray(img_array)

                    if use_hq:
                        alpha = img.getchannel('A')
                        img_rgb = img.convert('RGB')
                        img_p = img_rgb.convert('P', palette=Image.Palette.ADAPTIVE, colors=255)
                        mask = Image.eval(alpha, lambda a: 255 if a <= 128 else 0)
                        img_p.paste(255, mask)
                        img_p.info['transparency'] = 255
                        images.append(img_p)
                    else:
                        images.append(img)

                    # Rotate the CAMERA, not the molecule
                    self._orbit_camera(
                        plotter, axis_idx, step, i + 1, initial_pos_coord, initial_cam_up
                    )

                images[0].save(
                    save_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=frame_duration,
                    loop=0,
                    disposal=2
                )
            else:
                plotter.open_gif(save_path)
                for i in range(frames):
                    plotter.write_frame()
                    self._orbit_camera(
                        plotter, axis_idx, step, i + 1, initial_pos_coord, initial_cam_up
                    )

                if hasattr(plotter, 'mwriter') and plotter.mwriter is not None:
                    plotter.mwriter.close()

            self.context.show_status_message(f"GIF saved to: {save_path}", 5000)
            self.accept()

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.exception("Failed to generate GIF")
            QMessageBox.critical(self, "Error", f"Failed to generate GIF:\n{e}")
        finally:
            if use_hq:
                try:
                    if getattr(plotter, 'render_window', None) is not None:
                        plotter.disable_anti_aliasing()
                except Exception as aa_err:  # pylint: disable=broad-exception-caught
                    logger.warning("Failed to disable anti-aliasing: %s", aa_err)

            # Restore the camera state cleanly (including up-vector for view rotations)
            plotter.camera_position = initial_cpos
            plotter.camera.up = initial_cam_up
            self.context.refresh_3d_view()

    def _orbit_camera(
        self,
        plotter,
        axis_idx,
        step,
        frame_idx=1,
        initial_pos=None,
        initial_up=None,
    ):
        """
        Orbits the camera by 'step * frame_idx' degrees around the selected axis index
        relative to initial_pos/initial_up, updating VTK positions directly.
        """
        cam = plotter.camera

        # Ensure all coordinates are explicitly converted to NumPy arrays
        # This handles the tuple vs float/array mismatch
        fp = np.array(cam.focal_point, dtype=float)

        # Use initial_pos if provided, otherwise current cam.position
        pos = np.array(initial_pos if initial_pos is not None else cam.position, dtype=float)
        up = np.array(initial_up if initial_up is not None else cam.up, dtype=float)

        # Now vector subtraction is valid because both sides are NumPy arrays
        view_vec = pos - fp

        # Prevent division by zero
        eps = 1e-8
        view_norm = np.linalg.norm(view_vec)
        up_norm = np.linalg.norm(up)
        if view_norm < eps or up_norm < eps:
            return

        # Calculate local axes (camera-based)
        view_dir = view_vec / view_norm
        up_dir = up / up_norm
        right_dir = np.cross(up_dir, view_dir)

        # Determine the axis of rotation
        if axis_idx == 0:   # Z Axis (Global)
            axis = np.array([0.0, 0.0, 1.0])
        elif axis_idx == 1: # X Axis (Global)
            axis = np.array([1.0, 0.0, 0.0])
        elif axis_idx == 2: # Y Axis (Global)
            axis = np.array([0.0, 1.0, 0.0])
        elif axis_idx == 3: # Roll (Spin) -> line of sight
            axis = view_dir
        elif axis_idx == 4: # Elevation (Pitch) -> right vector
            axis = right_dir
        elif axis_idx == 5: # Azimuth (Yaw) -> up vector
            axis = up_dir
        else:
            return

        # Normalize the rotation axis
        axis_norm = np.linalg.norm(axis)
        if axis_norm < eps:
            return
        axis = axis / axis_norm

        # Calculate absolute rotation angle in radians
        theta = math.radians(-step * frame_idx)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # Rodrigues' rotation formula
        # Rotates vector 'v' around axis 'k' by 'theta'
        def rotate_vec(v, k):
            return (
                v * cos_t
                + np.cross(k, v) * sin_t
                + k * np.dot(k, v) * (1 - cos_t)
            )

        # Rotate both the view vector and the up vector simultaneously
        # (This ensures the 90-degree orthogonal relationship is maintained)
        new_view_vec = rotate_vec(view_vec, axis)
        new_up = rotate_vec(up, axis)

        # Update via VTK native setters to bypass PyVista's internal auto-reset triggers
        cam.SetPosition(*(fp + new_view_vec))
        cam.SetViewUp(*new_up)

        # Manually reset clipping range before rendering
        plotter.renderer.ResetCameraClippingRange()
        plotter.render()

        # Process paint events to allow macOS Cocoa backbuffer to update/swap buffers
        QCoreApplication.processEvents()
