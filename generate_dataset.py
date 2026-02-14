import pandas as pd
import numpy as np
import os
import argparse
import h5py
from scipy.spatial.transform import Rotation

# 17-joint kinematic parent map
PARENT_MAP = {
    13: 12,
    16: 13,
    15: 16,
    14: 15,  # Spine
    0: 15,
    1: 0,
    2: 1,  # Right Arm
    3: 15,
    4: 3,
    5: 4,  # Left Arm
    6: 12,
    7: 6,
    8: 7,  # Right Leg
    9: 12,
    10: 9,
    11: 10,  # Left Leg
}


def get_joint_headers(num_joints):
    headers = []
    for i in range(num_joints):
        headers.extend(
            [
                f"joint{i}_x",
                f"joint{i}_y",
                f"joint{i}_z",
                f"joint{i}_yaw",
                f"joint{i}_pitch",
                f"joint{i}_roll",
                f"joint{i}_conf",
            ]
        )
    return headers


def get_euler_angles_from_points(child_pos, parent_pos, ref_pos):
    """
    Derives a 3D coordinate frame to ensure non-zero roll.
    Uses the bone vector as one axis and a reference (e.g. pelvis) to define the plane.
    """
    # 1. Primary bone vector (Target Z-axis)
    v_bone = (child_pos - parent_pos).astype(np.float64)
    norm_v = np.linalg.norm(v_bone)
    if norm_v < 1e-6:
        return [0.0, 0.0, 0.0]
    z_axis = v_bone / norm_v

    # 2. Reference vector to define 'Up' (Ensures non-zero roll)
    # Using the vector from parent to pelvis to create a local plane
    v_ref = (ref_pos - parent_pos).astype(np.float64)
    norm_ref = np.linalg.norm(v_ref)
    if norm_ref < 1e-6:
        # Fallback to global up if joint is at the root
        up = np.array([0.0, 0.0, 1.0])
    else:
        up = v_ref / norm_ref

    # 3. Construct Orthogonal Basis (Gram-Schmidt)
    # Image of Gram-Schmidt orthonormalization process for coordinate frames
    x_axis = np.cross(up, z_axis)
    norm_x = np.linalg.norm(x_axis)

    if norm_x < 1e-6:  # Bone is parallel to reference
        # Shift reference to avoid gimbal lock in calculation
        x_axis = np.cross(np.array([1.0, 0.0, 0.0]), z_axis)
        x_axis /= np.linalg.norm(x_axis) + 1e-6
    else:
        x_axis /= norm_x

    y_axis = np.cross(z_axis, x_axis)

    # 4. Construct Rotation Matrix and convert to Euler
    # Columns are our local coordinate axes
    rot_matrix = np.stack([x_axis, y_axis, z_axis], axis=1)

    try:
        rot = Rotation.from_matrix(rot_matrix)
        # Using ZYX convention (Yaw, Pitch, Roll)
        euler = rot.as_euler("zyx", degrees=True)
        return [euler[0], euler[1], euler[2]]
    except:
        return [0.0, 0.0, 0.0]


def main():
    parser = argparse.ArgumentParser(
        description="Generate Dataset from Positions only."
    )
    parser.add_argument("--data_path", type=str, default="Data")
    parser.add_argument("--keypose_csv", type=str, default="assets/keyposes.csv")
    parser.add_argument("--output_dir", type=str, default="Datasets")
    parser.add_argument("--num_takes", type=int, default=2)
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--downsample_rate", type=int, default=5)
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir, f"N_{args.n}_Takes_{args.num_takes}.csv"
    )

    keypose_df = (
        pd.read_csv(args.keypose_csv)
        .melt(
            id_vars=["subject", "take"], var_name="class_index", value_name="frame_idx"
        )
        .dropna(subset=["frame_idx"])
    )
    keypose_df["frame_idx"] = keypose_df["frame_idx"].astype(int)

    all_rows = []

    for sub in args.subjects:
        print(f"[INFO] Processing Subject {sub}")
        subj_dir = os.path.join(args.data_path, f"Subject{sub}")

        for take_num in range(1, args.num_takes + 1):
            pos_path = os.path.join(subj_dir, f"MOCAP_3D_{take_num}.mat")
            if not os.path.exists(pos_path):
                continue

            with h5py.File(pos_path, "r") as f:
                mocap_pos = np.transpose(f["POSE"], (2, 1, 0))[:: args.downsample_rate]
                num_frames = mocap_pos.shape[0]
                is_labeled = np.zeros(num_frames, dtype=bool)
                relevant_labels = keypose_df[
                    (keypose_df["subject"] == sub) & (keypose_df["take"] == take_num)
                ]

                def process_frame(f_idx):
                    frame_feats = []
                    pelvis_p = mocap_pos[f_idx, 12, :3]
                    for j in range(17):
                        pos = mocap_pos[f_idx, j, :3] - pelvis_p
                        pos = [np.round(coord, 2) for coord in pos]
                        conf = mocap_pos[f_idx, j, 3]

                        # Derive Euler from kinematic parent
                        if j in PARENT_MAP:
                            euler = get_euler_angles_from_points(
                                mocap_pos[f_idx, j, :3],
                                mocap_pos[f_idx, PARENT_MAP[j], :3],
                                pelvis_p,
                            )
                            euler = [np.round(angle, 2) for angle in euler]
                        else:
                            euler = [0.0, 0.0, 0.0]

                        frame_feats.extend([*pos, *euler, conf])
                    return frame_feats

                for _, row in relevant_labels.iterrows():
                    c = int(row["frame_idx"] // args.downsample_rate)
                    s, e = max(0, c - args.n), min(num_frames, c + args.n)
                    is_labeled[s:e] = True
                    for i in range(s, e):
                        all_rows.append(
                            process_frame(i)
                            + [int(row["class_index"]) + 1, sub, take_num]
                        )

    pd.DataFrame(
        all_rows, columns=get_joint_headers(17) + ["Label", "Subject", "Take"]
    ).to_csv(output_path, index=False)
    print(f"[SUCCESS] Saved to {output_path}")


if __name__ == "__main__":
    main()
