"""Download the robot model (``uv run setup-assets``).

We fetch the Booster K1 from Booster Robotics' official asset repository,
github.com/BoosterRobotics/booster_assets (BSD-3-Clause — the license file
lands next to the meshes). A sparse, blob-filtered git clone downloads only
the K1 folder (~21 MB), not the other robots or the motion data.

Afterwards our scene (assets/scene_teleop.xml) is copied *into* the model
folder — see the comment at the top of that file for why it must sit next to
K1_22dof.xml. Re-running is safe and also refreshes the copied scene.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

from teleop import ASSETS_DIR

BOOSTER_GIT = "https://github.com/BoosterRobotics/booster_assets.git"
MODEL_DIR = ASSETS_DIR / "booster_k1"
SCENE_SRC = ASSETS_DIR / "scene_teleop.xml"
SCENE_DST = MODEL_DIR / "scene_teleop.xml"


def main() -> None:
    if (MODEL_DIR / "K1_22dof.xml").exists():
        print(f"model already present: {MODEL_DIR}")
    else:
        download()

    shutil.copyfile(SCENE_SRC, SCENE_DST)  # must live NEXT TO K1_22dof.xml (meshdir rule)
    print(f"scene installed: {SCENE_DST}")
    verify()


def download() -> None:
    tmp = ASSETS_DIR / ".booster_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    ASSETS_DIR.mkdir(exist_ok=True)
    print("downloading Booster K1 from booster_assets (sparse clone, ~21 MB)...")
    try:
        run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             BOOSTER_GIT, str(tmp)])
        run(["git", "-C", str(tmp), "sparse-checkout", "set", "robots/K1"])
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        shutil.rmtree(tmp, ignore_errors=True)
        sys.exit(
            f"download failed ({err}).\nManual fallback: clone "
            f"{BOOSTER_GIT} anywhere and copy its robots/K1/ into {MODEL_DIR}/"
        )
    shutil.move(str(tmp / "robots" / "K1"), str(MODEL_DIR))
    if (tmp / "LICENSE").exists():  # keep Booster's BSD-3 notice with the files
        shutil.copyfile(tmp / "LICENSE", MODEL_DIR / "LICENSE")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"model installed: {MODEL_DIR} (BSD-3-Clause, see its LICENSE file)")


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def verify() -> None:
    """Load the scene once and show the student what arrived."""
    from teleop.robot import SPECS, Robot  # late import: needs the files we just placed

    robot = Robot(SPECS["k1"])
    robot.reset()
    print(f"\nloaded OK: {robot.spec.scene_path().name}")
    print(f"  qpos size (nq): {robot.model.nq}   dof size (nv): {robot.model.nv}")
    for arm in robot.arms.values():
        hand = robot.ee_pose(arm)
        print(f"  {arm.name:>5} arm: {len(arm.joint_names)} joints (position-only IK), "
              f"palm at [{hand.pos[0]:+.2f} {hand.pos[1]:+.2f} {hand.pos[2]:+.2f}] m")
    print("\nnext:  uv run test-in-sim")


if __name__ == "__main__":
    main()
