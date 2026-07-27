"""Live pygame monitor: camera feed + simulated calibration view + coupled
dead-zone boundary trace/replay, all in one window -- for bench-testing
on a dev machine (see arm_hardware.Camera's "usb" backend) without
switching between separate tools.

This fuses manual_test/trace_boundary_gui.py's functionality in (boundary
trace/save/replay + the camera-free servo2_offset_deg quick-fix), so it is
NOT purely observational any more: it releases both joints' torque on
startup (ready to hand-trace immediately, same as that tool always did)
and re-syncs+re-locks on exit (_resync_and_relock). Only 'r' (replay)
actively commands motion; camera.capture_gray()/detector.detect() and the
tag-overlay/homography math are unchanged and still read-only.

Left panel: the live camera frame (whatever detector.detect() actually
sees), with every detected AprilTag marked (a circle at its pixel center +
its tag_id) -- ALL detected tags are shown, not just the ones calib.json
happens to name (corner/end-effector/card), so a mislabeled or stray tag
shows up immediately instead of being silently ignored.

Right panel: the reachable-area-shaded simulated view
manual_test/scan_area_gui.py uses (arm_core.ik_solve over a grid, computed
once at startup) + the yellow scan-area rectangle (arm_core.calib_scan_area)
+ the arm's REAL current pose (read from the servo encoders) + one dot per
detected tag, converted from its pixel center through calib.json's current
homography (arm_core.apply_homography) into workspace mm -- skipped if
homography hasn't been computed yet (see main.py homography), independent
of the camera panel's raw pixel-space markers -- PLUS, from
trace_boundary_gui.py: the live boundary trace while recording ('b') and a
ghost arm at the planned/commanded pose while replaying ('r').

Boundary trace/replay/save ('b'/'r'/'s') need calib.json's joint_limits_deg
to already have joint1/joint2's own independent ranges (see `main.py
set-joint-limits`) -- unlike that tool, THIS one doesn't refuse to start
without it (the camera/tag/simulated view is still useful on its own), it
just declines those three specific keys with an on-screen message until
that's been done. 'k'/shift+k (the servo2_offset_deg quick-fix) and 'c'
(clear trace) don't need it and are always available.

'j' toggles a second, mutually-exclusive mode -- LOCKED (torque enabled)
instead of the default UNLOCKED (torque released, ready to hand-trace).
While locked, arrow keys jog the arm (nudge_workspace, rotated by the
yellow scan-area's own tilt -- same convention fixed_path_scan/path_gui.py
uses) and '1'/'2' record the current jogged position as two corners of the
CARD's rectangle -- a manual fallback for arm_core.detect_card_rect() when
the camera can't reliably auto-detect the card's tag (e.g. focus issues).
'm' saves those two corners to calib.json's card.manual_corner_a_mm/
manual_corner_b_mm (see card_core.sub_rect_from_corners for how they
become a rectangle -- rotation inherited from the scan area, same as
fixed_path_scan). The manually-taught corners are drawn on the right panel
in a distinct color right next to the camera-detected tag positions
(_convert_tag_positions's dots), so a mismatch between "where I jogged to"
and "where the camera thinks the tag is" is visible at a glance. Can't
toggle lock state while recording or replaying (mutually exclusive, same
as those two are with each other).

Camera capture + AprilTag detection is far more expensive than reading an
encoder, so it's polled on its own (slower) interval independent of the
render/encoder-poll rate -- see CAMERA_POLL_INTERVAL_S.
"""

import argparse
import time
from dataclasses import dataclass

import numpy as np
import pygame

import arm_core as core
import arm_hardware as hw
import jog_controller as jc

import card_core as cc

FPS = 60
ENCODER_POLL_INTERVAL_S = 0.05
CAMERA_POLL_INTERVAL_S = 0.2
SHADING_STEP_PX = 4
CAM_PANEL_W = 480
CAM_MARGIN_PX = 20
STEP_MM = 5.0  # fixed jog step per arrow-key press while locked -- same as fixed_path_scan/path_gui.py
SCREENSHOT_PATH = "camera_view.png"
BOUNDARY_SCREENSHOT_PATH = "joint_limits_trace.png"
MAX_TAG_ROWS = 6

BG = (28, 30, 40)
REACHABLE_C = (40, 62, 46)
LINK1_C = (80, 160, 255)
LINK2_C = (255, 120, 55)
JOINT_C = (220, 220, 220)
EE_C = (55, 215, 95)
BASE_C = (200, 100, 55)
SCAN_AREA_C = (255, 210, 60)
TAG_MARK_C = (255, 230, 80)
TAG_MM_C = (120, 200, 255)
MANUAL_C = (255, 140, 60)
BOUNDARY_C = (255, 210, 60)
GHOST_C = (170, 90, 220)
TEXT_C = (200, 210, 225)
LABEL_C = (110, 130, 155)
HIGHLIGHT = (255, 220, 80)
WARN_C = (220, 80, 80)
OK_C = (80, 220, 130)


@dataclass
class Layout:
    """Screen-space layout centered on the arm's base, sized to its full
    reach circle -- same convention as manual_test/scan_area_gui.py's and
    manual_test/trace_boundary_gui.py's Layout, copied here rather than
    shared (this folder is self-contained, see its module docstrings for
    why)."""
    base_x: float
    base_y: float
    max_r_mm: float
    scale: float = 1.0
    margin_px: int = 40
    panel_w: int = 280

    def __post_init__(self):
        canvas_px = 560
        self.scale = canvas_px / (2 * self.max_r_mm)

    @property
    def win_w(self) -> int:
        return int(2 * self.max_r_mm * self.scale) + 2 * self.margin_px + self.panel_w

    @property
    def win_h(self) -> int:
        return int(2 * self.max_r_mm * self.scale) + 2 * self.margin_px

    def ws2s(self, wx: float, wy: float) -> tuple:
        cx = self.margin_px + self.max_r_mm * self.scale
        cy = self.margin_px + self.max_r_mm * self.scale
        return (int(cx + (wx - self.base_x) * self.scale),
                int(cy - (wy - self.base_y) * self.scale))

    def s2ws(self, px: float, py: float) -> tuple:
        cx = self.margin_px + self.max_r_mm * self.scale
        cy = self.margin_px + self.max_r_mm * self.scale
        return ((px - cx) / self.scale + self.base_x,
                self.base_y - (py - cy) / self.scale)


def _compute_reachable_surface(layout: Layout, params, joint_limits) -> pygame.Surface:
    """Precomputed once at startup -- see manual_test/scan_area_gui.py's
    twin of this function for why a live per-frame recompute isn't needed."""
    surf = pygame.Surface((layout.win_w, layout.win_h))
    surf.fill(BG)
    for py in range(0, layout.win_h, SHADING_STEP_PX):
        for px in range(0, layout.win_w, SHADING_STEP_PX):
            wx, wy = layout.s2ws(px, py)
            if core.ik_solve(params, wx, wy, joint_limits=joint_limits).reachable:
                pygame.draw.rect(surf, REACHABLE_C, (px, py, SHADING_STEP_PX, SHADING_STEP_PX))
    return surf


def _looks_uncalibrated(params: core.ArmParams) -> bool:
    """True if calib.json's kinematics are still the untouched nominal/CAD
    defaults -- see card_gui.py's identical helper."""
    nominal = core.ArmParams.nominal()
    return (params.L1 == nominal.L1 and params.L2 == nominal.L2 and
            params.base_x == nominal.base_x and params.base_y == nominal.base_y and
            params.servo1_offset_deg == nominal.servo1_offset_deg and
            params.servo2_offset_deg == nominal.servo2_offset_deg)


def _gray_to_surface(gray: np.ndarray) -> pygame.Surface:
    """A capture_gray() grayscale frame (numpy, shape (h, w)) -> a pygame
    Surface. pygame.surfarray.make_surface wants (w, h, 3), so this
    replicates the single channel to RGB and swaps the first two axes."""
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))


def _scale_to_width(surf: pygame.Surface, target_w: int) -> pygame.Surface:
    w, h = surf.get_size()
    target_h = max(1, int(target_w * h / w))
    return pygame.transform.smoothscale(surf, (target_w, target_h))


def _convert_tag_positions(detections: dict, H) -> dict:
    """tag_id -> (mm_x, mm_y) for EVERY detected tag, via
    arm_core.apply_homography -- deliberately unfiltered (a mislabeled or
    stray tag shows up here too, not just the ones calib.json happens to
    name as corner/ee/card tags). Empty dict if H is None (homography not
    yet configured, see main.py homography) -- the camera panel's raw
    pixel-space markers are unaffected either way, only this mm-space
    overlay is skipped."""
    if H is None:
        return {}
    return {tag_id: core.apply_homography(H, tuple(det.center)) for tag_id, det in detections.items()}


# ── from manual_test/trace_boundary_gui.py, unchanged ───────────────────

def _resync_and_relock(servos):
    """Sync goal to current position before re-enabling torque, so the
    servo doesn't snap toward a stale old target."""
    for joint in ("joint1", "joint2"):
        try:
            angle = servos.get_present_deg(joint)
            servos.set_target_deg(joint, angle)
        except Exception:
            pass
    for joint in ("joint1", "joint2"):
        servos.set_torque_enabled(joint, True)


def _build_replay_controller(servos, calib):
    """Same ArmController every other tool uses, but with the coupled-
    boundary polygon check dropped: replay's whole job is to visit that
    polygon's own vertices, which sit exactly ON its edge -- a case
    arm_core._point_in_polygon documents as genuinely ambiguous. The
    independent joint1/joint2 ranges (hardware-backed) are still enforced."""
    controller = jc.build_controller(servos, calib)
    if controller.joint_limits is not None:
        controller.joint_limits = {**controller.joint_limits, "coupled_boundary": []}
    return controller


def _replay_waypoints(boundary):
    """The saved boundary, walked in order and back to the first vertex."""
    return list(boundary) + [boundary[0]]


def main():
    parser = argparse.ArgumentParser(
        description="camera + calibration view, with coupled dead-zone boundary trace/replay")
    parser.add_argument(
        "--elbow-ref-deg", type=float, default=90.0,
        help="the L1-L2 angle you'll fold the arm to for the 'k' quick-fix key, "
             "as read by a protractor/set-square at the elbow -- 180=fully "
             "straight, 0=fully folded (default: 90.0)")
    args = parser.parse_args()

    calib = core.load_calib()
    hw_cfg = core.calib_hardware_config(calib)
    card_cfg = core.calib_card_config(calib)
    params = core.calib_arm_params(calib)
    limits = core.calib_joint_limits(calib)
    if limits is None:
        print("NOTE: calib.json has no joint_limits_deg yet -- 'b'/'r'/'s' (boundary "
              "trace/replay/save) are disabled until you run `main.py set-joint-limits` "
              "once. The camera/tag/simulated view below still works.")

    arm_hw = hw.ArmHardware(hw_cfg.servo_port, hw_cfg.joint_ids,
                             camera_backend=hw_cfg.camera_backend,
                             usb_camera_index=hw_cfg.usb_camera_index)
    arm_hw.connect()
    controller = _build_replay_controller(arm_hw.servos, calib)
    for joint in ("joint1", "joint2"):
        arm_hw.servos.set_torque_enabled(joint, False)

    uncalibrated = _looks_uncalibrated(params)
    H = calib["homography"].get("H")
    H = np.array(H) if H is not None else None

    print("computing reachable-area map...")
    max_r = params.L1 + params.L2
    layout = Layout(base_x=params.base_x, base_y=params.base_y, max_r_mm=max_r)
    reachable_surface = _compute_reachable_surface(layout, params, limits)
    print("done")

    # first_frame = arm_hw.camera.capture_gray()
    first_frame = np.zeros((480, 640), dtype=np.uint8)  # placeholder for sizing the window before the first camera poll
    cam_h, cam_w = first_frame.shape[:2]
    cam_panel_h = int(CAM_PANEL_W * cam_h / cam_w)
    cam_scale = CAM_PANEL_W / cam_w

    win_w = CAM_PANEL_W + CAM_MARGIN_PX + layout.win_w
    win_h = max(cam_panel_h, layout.win_h)

    pygame.init()
    pygame.display.set_caption("2R Arm -- camera + calibration view")
    screen = pygame.display.set_mode((win_w, win_h))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo,consolas,monospace", 18)
    sfont = pygame.font.SysFont("menlo,consolas,monospace", 13)

    real_s1 = arm_hw.servos.get_present_deg("joint1")
    real_s2 = arm_hw.servos.get_present_deg("joint2")
    last_encoder_poll = 0.0

    cached_frame = first_frame
    cached_detections = {}
    last_camera_poll = 0.0

    screenshot_pending = False
    boundary_screenshot_pending = False
    msg = ""
    msg_until = 0.0

    boundary_trace: list = []  # (joint1_deg, joint2_deg), in traced order
    recording = False
    replaying = False
    replay_queue: list = []

    jogging = False  # locked (torque on, arrow-key jog + '1'/'2') vs default unlocked
    workspace_target = core.fk_from_servo_angles(params, real_s1, real_s2)
    manual_corner_a = card_cfg.manual_corner_a_mm
    manual_corner_b = card_cfg.manual_corner_b_mm

    # (center_x, center_y, width, height, rotation_deg) -- read-only here,
    # configure/rotate it with manual_test/scan_area_gui.py. Fixed for the
    # whole run (this tool never edits it), read once rather than every
    # frame; jog direction and the manual card rectangle's rotation both
    # come from scan_rot.
    scan_area = core.calib_scan_area(calib)
    scan_rot = scan_area[4]

    running = True
    try:
        while running:
            now = time.monotonic()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_p:
                        screenshot_pending = True
                    elif event.key == pygame.K_c:
                        boundary_trace = []
                        recording = False
                    elif event.key == pygame.K_j:
                        if recording:
                            msg = "stop recording ('b') before toggling lock"
                            msg_until = now + 3.0
                        elif replaying:
                            msg = "stop the replay ('r') before toggling lock"
                            msg_until = now + 3.0
                        elif jogging:
                            arm_hw.servos.set_torque_enabled("joint1", False)
                            arm_hw.servos.set_torque_enabled("joint2", False)
                            jogging = False
                            msg = "unlocked (hand-trace mode)"
                            msg_until = now + 2.0
                        else:
                            arm_hw.servos.set_torque_enabled("joint1", True)
                            arm_hw.servos.set_torque_enabled("joint2", True)
                            workspace_target = core.fk_from_servo_angles(params, *controller.commanded_deg)
                            jogging = True
                            msg = "locked (arrow keys jog, 1/2 record card corners)"
                            msg_until = now + 3.0
                    elif event.key == pygame.K_m:
                        if manual_corner_a is None or manual_corner_b is None:
                            msg = "record both corners ('1' and '2', while locked) before saving"
                            msg_until = now + 3.0
                        else:
                            calib.setdefault("card", {})
                            calib["card"]["manual_corner_a_mm"] = list(manual_corner_a)
                            calib["card"]["manual_corner_b_mm"] = list(manual_corner_b)
                            core.save_calib(calib)
                            msg = "saved manual card corners to calib.json"
                            msg_until = now + 3.0
                    elif jogging and event.key == pygame.K_1:
                        manual_corner_a = workspace_target
                        msg = f"corner A recorded: ({workspace_target[0]:.1f}, {workspace_target[1]:.1f}) mm"
                        msg_until = now + 3.0
                    elif jogging and event.key == pygame.K_2:
                        manual_corner_b = workspace_target
                        msg = f"corner B recorded: ({workspace_target[0]:.1f}, {workspace_target[1]:.1f}) mm"
                        msg_until = now + 3.0
                    elif jogging and event.key == pygame.K_UP:
                        dx, dy = core.rotate_vector(0.0, STEP_MM, scan_rot)
                        new_t = controller.nudge_workspace(dx, dy, workspace_target)
                        workspace_target = new_t if new_t else workspace_target
                    elif jogging and event.key == pygame.K_DOWN:
                        dx, dy = core.rotate_vector(0.0, -STEP_MM, scan_rot)
                        new_t = controller.nudge_workspace(dx, dy, workspace_target)
                        workspace_target = new_t if new_t else workspace_target
                    elif jogging and event.key == pygame.K_LEFT:
                        dx, dy = core.rotate_vector(-STEP_MM, 0.0, scan_rot)
                        new_t = controller.nudge_workspace(dx, dy, workspace_target)
                        workspace_target = new_t if new_t else workspace_target
                    elif jogging and event.key == pygame.K_RIGHT:
                        dx, dy = core.rotate_vector(STEP_MM, 0.0, scan_rot)
                        new_t = controller.nudge_workspace(dx, dy, workspace_target)
                        workspace_target = new_t if new_t else workspace_target
                    elif event.key == pygame.K_k:
                        flip = bool(event.mod & pygame.KMOD_SHIFT)
                        old = calib["kinematics"]["servo2_offset_deg"]
                        new = core.servo2_offset_from_known_elbow_angle(
                            real_s2, params.servo2_dir, args.elbow_ref_deg, flip=flip)
                        calib["kinematics"]["servo2_offset_deg"] = new
                        core.save_calib(calib)
                        params = core.calib_arm_params(calib)
                        other_key = "k" if flip else "shift+k"
                        msg = (f"servo2_offset_deg: {old:.2f} -> {new:.2f} (saved) -- "
                               f"if wrong/mirrored, try {other_key}")
                        msg_until = now + 6.0
                    elif event.key == pygame.K_b:
                        if limits is None:
                            msg = "run main.py set-joint-limits first"
                            msg_until = now + 3.0
                        elif replaying:
                            msg = "stop the replay ('r') before recording"
                            msg_until = now + 3.0
                        else:
                            if not recording:
                                boundary_trace = []
                                arm_hw.servos.set_torque_enabled("joint1", False)
                                arm_hw.servos.set_torque_enabled("joint2", False)
                            recording = not recording
                    elif event.key == pygame.K_r:
                        if limits is None:
                            msg = "run main.py set-joint-limits first"
                            msg_until = now + 3.0
                        elif recording:
                            msg = "stop recording ('b') before replaying"
                            msg_until = now + 3.0
                        elif replaying:
                            replaying = False
                            replay_queue = []
                            msg = "replay stopped"
                            msg_until = now + 2.0
                        elif not limits["coupled_boundary"]:
                            msg = "no saved coupled_boundary to replay yet -- trace and save one first"
                            msg_until = now + 3.0
                        else:
                            arm_hw.servos.set_torque_enabled("joint1", True)
                            arm_hw.servos.set_torque_enabled("joint2", True)
                            replay_queue = _replay_waypoints(limits["coupled_boundary"])
                            controller.set_joint_goal(*replay_queue.pop(0))
                            replaying = True
                    elif event.key in (pygame.K_s, pygame.K_RETURN):
                        if limits is None:
                            msg = "run main.py set-joint-limits first"
                            msg_until = now + 3.0
                        elif len(boundary_trace) < 3:
                            msg = "need >=3 vertices to form a closed loop -- keep tracing"
                            msg_until = now + 3.0
                        else:
                            j1_lo = min([limits["joint1"][0]] + [v[0] for v in boundary_trace])
                            j1_hi = max([limits["joint1"][1]] + [v[0] for v in boundary_trace])
                            j2_lo = min([limits["joint2"][0]] + [v[1] for v in boundary_trace])
                            j2_hi = max([limits["joint2"][1]] + [v[1] for v in boundary_trace])
                            calib["joint_limits_deg"] = {
                                "joint1": [round(j1_lo, 2), round(j1_hi, 2)],
                                "joint2": [round(j2_lo, 2), round(j2_hi, 2)],
                                "coupled_boundary": [{"joint1": round(j1, 2), "joint2": round(j2, 2)}
                                                     for j1, j2 in boundary_trace],
                            }
                            core.save_calib(calib)
                            limits = core.calib_joint_limits(calib)
                            controller.joint_limits = {**limits, "coupled_boundary": []}
                            boundary_screenshot_pending = True
                            msg = (f"saved {len(boundary_trace)}-vertex boundary to "
                                   f"calib.json + {BOUNDARY_SCREENSHOT_PATH}")
                            msg_until = now + 4.0

            if now - last_encoder_poll >= ENCODER_POLL_INTERVAL_S:
                real_s1 = arm_hw.servos.get_present_deg("joint1")
                real_s2 = arm_hw.servos.get_present_deg("joint2")
                if recording:
                    boundary_trace.append((real_s1, real_s2))
                last_encoder_poll = now

            if now - last_camera_poll >= CAMERA_POLL_INTERVAL_S:
                cached_frame = arm_hw.camera.capture_gray()
                cached_detections = arm_hw.detector.detect(cached_frame)
                last_camera_poll = now

            if jogging or replaying:
                # Ticking is what actually drains the planner's queue into
                # set_target_deg calls (nudge_workspace/set_joint_goal only
                # plan a segment) -- jogging needs this exactly as much as
                # replaying does, it just has no replay_queue to advance.
                controller.tick()
                if replaying and not controller.is_moving:
                    if replay_queue:
                        controller.set_joint_goal(*replay_queue.pop(0))
                    else:
                        replaying = False
                        msg = "replay finished"
                        msg_until = now + 3.0

            screen.fill(BG)

            # ── left: camera feed + tag overlay ──
            cam_surface = _scale_to_width(_gray_to_surface(cached_frame), CAM_PANEL_W)
            screen.blit(cam_surface, (0, 0))
            for tag_id, det in cached_detections.items():
                px, py = det.center[0] * cam_scale, det.center[1] * cam_scale
                pygame.draw.circle(screen, TAG_MARK_C, (int(px), int(py)), 8, 2)
                screen.blit(sfont.render(str(tag_id), True, TAG_MARK_C), (int(px) + 10, int(py) - 8))

            # ── right: simulated calibration + boundary trace/replay view ──
            sim_surface = reachable_surface.copy()

            scan_points = [layout.ws2s(*c) for c in core.scan_area_corners(*scan_area)]
            pygame.draw.polygon(sim_surface, SCAN_AREA_C, scan_points, 2)

            if boundary_trace:
                loop = [layout.ws2s(*core.fk_from_servo_angles(params, j1, j2))
                        for j1, j2 in boundary_trace]
                if len(loop) >= 2:
                    pygame.draw.lines(sim_surface, BOUNDARY_C, True, loop, 2)
                else:
                    pygame.draw.circle(sim_surface, BOUNDARY_C, loop[0], 3)

            if replaying:
                cmd_s1, cmd_s2 = controller.commanded_deg
                g_elbow, g_ee = core.fk_joint_positions(params, cmd_s1, cmd_s2)
                p_base_g, p_elbow_g, p_ee_g = (layout.ws2s(params.base_x, params.base_y),
                                                layout.ws2s(*g_elbow), layout.ws2s(*g_ee))
                pygame.draw.line(sim_surface, GHOST_C, p_base_g, p_elbow_g, 3)
                pygame.draw.line(sim_surface, GHOST_C, p_elbow_g, p_ee_g, 2)
                pygame.draw.circle(sim_surface, GHOST_C, p_ee_g, 5, 1)

            elbow, ee = core.fk_joint_positions(params, real_s1, real_s2)
            p_base, p_elbow, p_ee = (layout.ws2s(params.base_x, params.base_y),
                                      layout.ws2s(*elbow), layout.ws2s(*ee))
            pygame.draw.line(sim_surface, LINK1_C, p_base, p_elbow, 6)
            pygame.draw.line(sim_surface, LINK2_C, p_elbow, p_ee, 5)
            pygame.draw.circle(sim_surface, JOINT_C, p_elbow, 7)
            pygame.draw.circle(sim_surface, EE_C, p_ee, 8)
            pygame.draw.circle(sim_surface, BASE_C, p_base, 9, 2)

            tag_mm = _convert_tag_positions(cached_detections, H)
            for tag_id, mm_xy in tag_mm.items():
                p = layout.ws2s(*mm_xy)
                pygame.draw.circle(sim_surface, TAG_MM_C, p, 5)
                sim_surface.blit(sfont.render(str(tag_id), True, TAG_MM_C), (p[0] + 8, p[1] - 8))

            # Manually-jogged card corners -- a distinct color right next
            # to the (blue) tag-detected dots above, so a mismatch between
            # "where I jogged to" and "where the camera thinks the tag is"
            # is visible at a glance.
            if manual_corner_a is not None:
                pygame.draw.circle(sim_surface, MANUAL_C, layout.ws2s(*manual_corner_a), 6, 2)
            if manual_corner_b is not None:
                pygame.draw.circle(sim_surface, MANUAL_C, layout.ws2s(*manual_corner_b), 6, 2)
            if manual_corner_a is not None and manual_corner_b is not None:
                manual_rect = cc.sub_rect_from_corners(scan_area, manual_corner_a, manual_corner_b)
                manual_points = [layout.ws2s(*c) for c in core.scan_area_corners(*manual_rect)]
                pygame.draw.polygon(sim_surface, MANUAL_C, manual_points, 1)

            px, py = int(2 * max_r * layout.scale) + layout.margin_px * 2, 28

            def row(text, dy, color=TEXT_C, f=None):
                nonlocal py
                sim_surface.blit((f or font).render(text, True, color), (px, py))
                py += dy

            row("CAMERA + CALIBRATION VIEW", 22, HIGHLIGHT)
            row(f"joint1={real_s1:6.1f}  joint2={real_s2:6.1f}", 16, TEXT_C, sfont)
            theta2_now = params.servo2_dir * (real_s2 - params.servo2_offset_deg)
            row(f"L1-L2 angle (sim): {180.0 - theta2_now:6.1f} deg", 22, TEXT_C, sfont)

            if uncalibrated:
                row("WARNING: kinematics look uncalibrated", 13, WARN_C, sfont)
                row("run main.py homography + calibrate first", 16, WARN_C, sfont)
            else:
                row(" ", 13)
                row(" ", 16)

            row("homography: configured" if H is not None else "homography: NOT configured",
                16, OK_C if H is not None else WARN_C, sfont)

            cx, cy, w, h, rot = scan_area
            row(f"scan area: {w:.0f}x{h:.0f}mm rot={rot:.1f}deg "
                f"center=({cx:.1f},{cy:.1f})", 16, LABEL_C, sfont)
            row(f"card tag_id={card_cfg.tag_id} size={card_cfg.width_mm:.1f}x"
                f"{card_cfg.height_mm:.1f}mm", 16, LABEL_C, sfont)

            row("RECORDING" if recording else "not recording", 16,
                OK_C if recording else LABEL_C, sfont)
            row(f"trace: {len(boundary_trace)} vertices", 16, LABEL_C, sfont)
            if limits is not None:
                saved_n = len(limits["coupled_boundary"])
                if replaying:
                    row(f"REPLAYING -- {len(replay_queue)} waypoints left", 16, WARN_C, sfont)
                else:
                    row(f"saved boundary: {saved_n} vertices" if saved_n else
                        "saved boundary: none yet", 16, LABEL_C, sfont)
            else:
                row("joint_limits_deg: not configured", 16, WARN_C, sfont)

            row("LOCKED (jog)" if jogging else "unlocked (hand-trace)", 16,
                WARN_C if jogging else OK_C, sfont)
            a_txt = f"({manual_corner_a[0]:.1f},{manual_corner_a[1]:.1f})" if manual_corner_a else "not set"
            b_txt = f"({manual_corner_b[0]:.1f},{manual_corner_b[1]:.1f})" if manual_corner_b else "not set"
            row(f"manual corner A: {a_txt}", 16, LABEL_C, sfont)
            row(f"manual corner B: {b_txt}", 16, LABEL_C, sfont)

            if msg and now < msg_until:
                row(msg, 22, OK_C, sfont)
            else:
                row(" ", 22)

            tag_ids = sorted(cached_detections)
            row(f"-- {len(tag_ids)} tag(s) detected --", 13, LABEL_C, sfont)
            for tag_id in tag_ids[:MAX_TAG_ROWS]:
                det = cached_detections[tag_id]
                if tag_id in tag_mm:
                    mm_xy = tag_mm[tag_id]
                    row(f"id={tag_id}: px=({det.center[0]:.0f},{det.center[1]:.0f}) "
                        f"-> mm=({mm_xy[0]:.1f},{mm_xy[1]:.1f})", 16, LABEL_C, sfont)
                else:
                    row(f"id={tag_id}: px=({det.center[0]:.0f},{det.center[1]:.0f}) -> mm=N/A",
                        16, LABEL_C, sfont)
            if len(tag_ids) > MAX_TAG_ROWS:
                row(f"+{len(tag_ids) - MAX_TAG_ROWS} more", 16, LABEL_C, sfont)

            row(" ", 8)
            row("-- Keys --", 13, LABEL_C, sfont)
            for line in ["p         screenshot",
                         "b         start/stop recording the boundary",
                         "c         clear trace, stop recording",
                         f"s/Enter   save boundary + {BOUNDARY_SCREENSHOT_PATH}",
                         "r         replay saved boundary (drives the arm!)",
                         f"k         fold elbow to {args.elbow_ref_deg:.0f}deg, fix",
                         "          servo2_offset_deg (no camera needed)",
                         "shift+k   same, other fold direction",
                         "j         toggle lock (jog mode) on/off",
                         "arrows    (locked only) jog 5mm/press",
                         "1 / 2     (locked only) record card corner A/B",
                         "m         save manual card corners to calib.json",
                         "q/ESC     quit"]:
                row(line, 15, LABEL_C, sfont)

            screen.blit(sim_surface, (CAM_PANEL_W + CAM_MARGIN_PX, 0))

            pygame.display.flip()
            if screenshot_pending:
                try:
                    pygame.image.save(screen, SCREENSHOT_PATH)
                    print(f"saved a screenshot to {SCREENSHOT_PATH}")
                except Exception as e:  # noqa: BLE001 -- a screenshot is optional, never fatal
                    print(f"WARNING: could not save {SCREENSHOT_PATH} ({e})")
                screenshot_pending = False
            if boundary_screenshot_pending:
                try:
                    pygame.image.save(screen, BOUNDARY_SCREENSHOT_PATH)
                    print(f"saved a screenshot to {BOUNDARY_SCREENSHOT_PATH}")
                except Exception as e:  # noqa: BLE001 -- a screenshot is optional, never fatal
                    print(f"WARNING: could not save {BOUNDARY_SCREENSHOT_PATH} ({e}) -- "
                          f"calib.json's data is unaffected")
                boundary_screenshot_pending = False
            clock.tick(FPS)
    finally:
        pygame.quit()
        _resync_and_relock(arm_hw.servos)
        arm_hw.close()


if __name__ == "__main__":
    main()
