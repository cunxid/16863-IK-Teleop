"""Download the robot model (``uv run setup-assets``).

We fetch the Unitree G1 from google-deepmind/mujoco_menagerie (BSD-3-Clause,
©Unitree Robotics — the license file lands next to the meshes). A sparse,
blob-filtered git clone downloads only the one robot folder (~38 MB), not the
whole zoo.

Afterwards our scene (assets/scene_teleop.xml) is copied *into* the model
folder — see the comment at the top of that file for why it must sit next to
g1.xml. Re-running is safe and also refreshes the copied scene.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

from teleop import ASSETS_DIR

MENAGERIE_GIT = "https://github.com/google-deepmind/mujoco_menagerie.git"
MODEL_DIR = ASSETS_DIR / "unitree_g1"
SCENE_SRC = ASSETS_DIR / "scene_teleop.xml"
SCENE_DST = MODEL_DIR / "scene_teleop.xml"


def main() -> None:
    if (MODEL_DIR / "g1.xml").exists():
        print(f"model already present: {MODEL_DIR}")
    else:
        download()

    shutil.copyfile(SCENE_SRC, SCENE_DST)  # must live NEXT TO g1.xml (meshdir rule)
    print(f"scene installed: {SCENE_DST}")
    verify()


def download() -> None:
    tmp = ASSETS_DIR / ".menagerie_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    ASSETS_DIR.mkdir(exist_ok=True)
    print("downloading Unitree G1 from mujoco_menagerie (sparse clone, ~38 MB)...")
    try:
        run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             MENAGERIE_GIT, str(tmp)])
        run(["git", "-C", str(tmp), "sparse-checkout", "set", "unitree_g1"])
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        shutil.rmtree(tmp, ignore_errors=True)
        sys.exit(
            f"download failed ({err}).\nManual fallback: clone "
            f"{MENAGERIE_GIT} anywhere and copy its unitree_g1/ into {ASSETS_DIR}/"
        )
    shutil.move(str(tmp / "unitree_g1"), str(MODEL_DIR))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"model installed: {MODEL_DIR} (BSD-3-Clause, see its LICENSE file)")


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def verify() -> None:
    """Load the scene once and show the student what arrived."""
    from teleop.robot import SPECS, Robot  # late import: needs the files we just placed

    robot = Robot(SPECS["g1"])
    robot.reset()
    print(f"\nloaded OK: {robot.spec.scene_path().name}")
    print(f"  qpos size (nq): {robot.model.nq}   dof size (nv): {robot.model.nv}")
    for arm in robot.arms.values():
        hand = robot.ee_pose(arm)
        print(f"  {arm.name:>5} arm: {len(arm.joint_names)} joints, "
              f"palm at [{hand.pos[0]:+.2f} {hand.pos[1]:+.2f} {hand.pos[2]:+.2f}] m")
    print("\nnext:  uv run test-in-sim")


if __name__ == "__main__":
    main()
