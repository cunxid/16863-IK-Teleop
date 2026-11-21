"""Download the robot models (``uv run setup-assets [--robot k1|t1|all]``).

We fetch the Booster robots from Booster Robotics' official asset repository,
github.com/BoosterRobotics/booster_assets (BSD-3-Clause — the license file
lands next to the meshes). A sparse, blob-filtered git clone downloads only
the folders we need (~20 MB per robot), not the other robots or the motion
data. *Which* folders — and every other robot-specific string — comes from
the ``RobotSpec`` objects in robot.py, so adding a robot there is enough to
teach this tool about it.

Afterwards each robot's scene (e.g. assets/scene_teleop_k1.xml) is copied
*into* its model folder — see the comment at the top of that file for why it
must sit next to the vendor XML. Re-running is safe and also refreshes the
copied scenes.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

from teleop import ASSETS_DIR
from teleop.robot import SPECS, RobotSpec

BOOSTER_GIT = "https://github.com/BoosterRobotics/booster_assets.git"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the robot model(s) and install the teleop scenes.")
    parser.add_argument("--robot", choices=[*sorted(SPECS), "all"], default="all",
                        help="which robot to set up (default: all of them)")
    args = parser.parse_args()
    specs = list(SPECS.values()) if args.robot == "all" else [SPECS[args.robot]]

    missing = [spec for spec in specs if not spec.model_path().exists()]
    for spec in specs:
        if spec not in missing:
            print(f"model already present: {spec.model_dir()}")
    if missing:
        download(missing)

    for spec in specs:
        # The scene must live NEXT TO the vendor XML (meshdir rule, lesson §2).
        shutil.copyfile(ASSETS_DIR / spec.scene_template, spec.scene_path())
        print(f"scene installed: {spec.scene_path()}")
        verify(spec)


def download(specs: list[RobotSpec]) -> None:
    tmp = ASSETS_DIR / ".booster_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    ASSETS_DIR.mkdir(exist_ok=True)
    names = ", ".join(spec.name for spec in specs)
    print(f"downloading {names} from booster_assets (sparse clone, ~20 MB per robot)...")
    try:
        run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             BOOSTER_GIT, str(tmp)])
        run(["git", "-C", str(tmp), "sparse-checkout", "set",
             *(spec.vendor_dir for spec in specs)])
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        shutil.rmtree(tmp, ignore_errors=True)
        sys.exit(
            f"download failed ({err}).\nManual fallback: clone "
            f"{BOOSTER_GIT} anywhere and copy "
            + "; ".join(f"its {spec.vendor_dir}/ into {spec.model_dir()}/" for spec in specs)
        )
    for spec in specs:
        shutil.move(str(tmp / spec.vendor_dir), str(spec.model_dir()))
        if (tmp / "LICENSE").exists():  # keep Booster's BSD-3 notice with the files
            shutil.copyfile(tmp / "LICENSE", spec.model_dir() / "LICENSE")
        print(f"model installed: {spec.model_dir()} (BSD-3-Clause, see its LICENSE file)")
    shutil.rmtree(tmp, ignore_errors=True)


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def verify(spec: RobotSpec) -> None:
    """Load the scene once and show the student what arrived."""
    from teleop.robot import Robot  # late import: needs the files we just placed

    robot = Robot(spec)
    robot.reset()
    print(f"\nloaded OK: {robot.spec.scene_path()}")
    print(f"  qpos size (nq): {robot.model.nq}   dof size (nv): {robot.model.nv}")
    for arm in robot.arms.values():
        hand = robot.ee_pose(arm)
        print(f"  {arm.name:>5} arm: {len(arm.joint_names)} joints (position-only IK), "
              f"palm at [{hand.pos[0]:+.2f} {hand.pos[1]:+.2f} {hand.pos[2]:+.2f}] m")
    print(f"\nnext:  uv run test-in-sim --robot-spec {spec.name}\n")


if __name__ == "__main__":
    main()
