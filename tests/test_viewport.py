import math
import os
import unittest
from pathlib import Path

from stemspace.project import DEFAULTS, Placement, Project, Track, default_placement
from stemspace.spatial import Camera, Projection, azimuth_of, direction, gain_radius, width_arc

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from stemspace.ui import Window
    from stemspace.viewport import overlap_badge, screen
except ModuleNotFoundError as exc:
    if exc.name != "PySide6":
        raise
    QApplication = None


class GeometryTest(unittest.TestCase):
    def test_projection_and_drag_ray_agree_around_the_scene(self):
        for camera in (Camera(), Camera(165, 14), Camera(-125, 81, 1.5)):
            projection = Projection(camera, 610, 430)
            for azimuth, elevation in ((-175, 12), (-60, 0), (0, 80), (45, 30), (172, 65)):
                point = direction(azimuth, elevation)
                displayed = projection.project(point)
                restored = projection.on_horizontal(displayed.x, displayed.y, point.z)
                self.assertIsNotNone(restored)
                self.assertAlmostEqual(restored.x, point.x)
                self.assertAlmostEqual(restored.y, point.y)
                self.assertAlmostEqual(azimuth_of(restored), azimuth)

    def test_stereo_spread_follows_azimuth_and_gain_does_not_move_the_marker(self):
        arc = width_arc(170, 45, 60)
        self.assertAlmostEqual(azimuth_of(arc[0]), 140)
        self.assertAlmostEqual(azimuth_of(arc[-1]), -160)
        self.assertTrue(all(math.isclose(p.z, arc[0].z) for p in arc))
        self.assertEqual(set(width_arc(42, 90, 72)), {direction(0, 90)})
        self.assertLess(gain_radius(-12), gain_radius(0))
        self.assertLess(gain_radius(0), gain_radius(6))


@unittest.skipIf(QApplication is None, "install the UI extra to test the viewport")
class ViewportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = Window()
        project = Project(Path("."), "fixture.wav", "", {}, "fixture.wav", "residual.wav", 48000, 48000,
                          [Track(name, name.capitalize(), f"{name}.wav", default_placement(name)) for name in DEFAULTS],
                          {"peak_error": 0})
        self.window.apply_project(project)
        self.window.show()
        self.app.processEvents()
        self.view = self.window.viewport
        self.project = project
        self.ride = next(i for i, t in enumerate(project.tracks) if t.id == "ride")
        self.front = next(i for i, t in enumerate(project.tracks) if t.id == "bass")

    def tearDown(self):
        self.window.dirty = False
        self.window.close()
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()

    def marker(self, i):
        return screen(self.view.markers(self.view.projection())[i]).toPoint()

    def click(self, point):
        QTest.mouseClick(self.view, Qt.MouseButton.LeftButton, pos=point)
        self.app.processEvents()

    def drag(self, start, end):
        QTest.mousePress(self.view, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(self.view, end)
        self.app.processEvents()
        QTest.mouseRelease(self.view, Qt.MouseButton.LeftButton, pos=end)
        self.app.processEvents()

    def test_click_selects_a_track_and_click_again_cycles_the_front_stack(self):
        self.click(self.marker(self.ride))
        self.assertEqual(self.window.tracks.currentRow(), self.ride)
        self.assertEqual(self.window.elevation.value(), 45)
        front = self.marker(self.front)
        names = []
        for _ in range(4):
            self.click(front)
            names.append(self.window.current_track().id)
        self.assertEqual(set(names), {"bass", "kick", "vocals", "other"})
        self.click(front)
        self.assertEqual(self.window.current_track().id, names[0])
        badge = overlap_badge(screen(self.view.markers(self.view.projection())[self.front])).center().toPoint()
        self.click(badge)
        picked = self.window.current_track().id
        self.click(badge)
        self.assertNotEqual(self.window.current_track().id, picked)
        self.assertIn(self.window.current_track().id, names)
        self.assertFalse(self.window.dirty)

    def test_drag_marker_and_height_handle_edit_different_axes(self):
        self.click(self.marker(self.ride))
        target = screen(self.view.projection().project(direction(165, 45))).toPoint()
        self.drag(self.marker(self.ride), target)
        after_yaw = self.project.tracks[self.ride].placement
        self.assertAlmostEqual(after_yaw.azimuth, 165, delta=2)
        self.assertEqual(after_yaw.elevation, 45)

        handle = self.view.height_handle(self.view.markers(self.view.projection()))
        self.assertIsNotNone(handle)
        self.drag(handle.toPoint(), handle.toPoint() - QPoint(0, 29))
        placement = self.project.tracks[self.ride].placement
        self.assertEqual(placement.azimuth, after_yaw.azimuth)
        self.assertGreater(placement.elevation, 50)
        self.assertEqual((placement.width, placement.gain_db), (0, 0))

    def test_camera_controls_do_not_edit_tracks(self):
        before = [t.placement for t in self.project.tracks]
        initial_marker = self.marker(self.ride)
        self.drag(QPoint(12, 24), QPoint(90, 66))
        self.assertNotEqual(self.view.camera, Camera())
        self.assertNotEqual(self.marker(self.ride), initial_marker)
        zoom = self.view.camera.zoom
        self.window.zoom_in.click()
        self.assertGreater(self.view.camera.zoom, zoom)
        self.window.reset_view.click()
        self.assertEqual(self.view.camera, Camera())
        self.assertEqual(self.marker(self.ride), initial_marker)
        self.assertEqual([t.placement for t in self.project.tracks], before)
        self.assertFalse(self.window.dirty)

    def test_width_gain_and_elevation_are_live_in_the_scene(self):
        self.window.tracks.setCurrentRow(self.ride)
        self.assertEqual(self.view.selected, self.ride)
        initial = self.marker(self.ride)
        images = [self.view.grab().toImage()]
        for control, value in ((self.window.width, 72), (self.window.gain, -12),
                               (self.window.elevation, 65)):
            control.setValue(value)
            self.app.processEvents()
            images.append(self.view.grab().toImage())
        for before, after in zip(images, images[1:]):
            self.assertNotEqual(before, after)
        self.assertNotEqual(self.marker(self.ride), initial)
        self.assertEqual(self.project.tracks[self.ride].placement, Placement(125, 65, 72, -12))


if __name__ == "__main__":
    unittest.main()
