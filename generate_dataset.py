import pandas as pd
import numpy as np
import os
import h5py

def get_joint_headers(num_joints):
    headers = []
    for i in range(num_joints):
        headers.extend([f'joint{i}_x', f'joint{i}_y', f'joint{i}_z', f'joint{i}_conf'])
    return headers

pelvis_center_idx = 12  # (unused in your snippet, keeping it)

def create_dataset_file(
    data_path="Data",
    keypose_csv="assets/keyposes.csv",
    output_dir="Datasets",
    num_takes=2,
    subjects=None,
    downsample_rate=5,
    n=20,
    seed=42
):
    """
    Generate simplified Keypose CSV without argparse.
    """
    if subjects is None:
        subjects = list(range(1, 11))

    if seed is not None:
        np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"N_{n}_Takes_{num_takes}.csv"
    output_path = os.path.join(output_dir, output_filename)

    raw_keypose_df = pd.read_csv(keypose_csv)

    keypose_df = raw_keypose_df.melt(
        id_vars=["subject", "take"],
        var_name="class_index",
        value_name="frame_idx"
    ).dropna(subset=["frame_idx"])

    keypose_df["frame_idx"] = keypose_df["frame_idx"].astype(int)

    all_rows = []
    num_joints = None  # will infer once we load first file

    for sub in subjects:
        print(f"[INFO] Processing Subject {sub}")
        subj_dir = os.path.join(data_path, f"Subject{sub}")

        for take_num in range(1, num_takes + 1):
            file_path = os.path.join(subj_dir, f"MOCAP_3D_{take_num}.mat")

            mat_file = h5py.File(file_path, "r")
            mocap_data = mat_file["POSE"]
            mocap_array = np.transpose(mocap_data, (2, 1, 0))
            mocap_array = mocap_array[::downsample_rate]

            if num_joints is None:
                num_joints = mocap_array.shape[1]

            num_frames = mocap_array.shape[0]
            is_labeled = np.zeros(num_frames, dtype=bool)

            relevant_labels = keypose_df[
                (keypose_df["subject"] == sub) & (keypose_df["take"] == take_num)
            ]

            # Add labeled windows
            for _, row in relevant_labels.iterrows():
                center_idx = int(row["frame_idx"] // downsample_rate)
                start, end = center_idx - n, center_idx + n

                start = max(start, 0)
                end = min(end, num_frames)

                is_labeled[start:end] = True

                # class_index is a column name from melt; ensure int conversion is safe
                new_label = int(row["class_index"]) + 1

                segment = mocap_array[start:end]
                for frame in segment:
                    flat_frame = frame.flatten().tolist()
                    all_rows.append(flat_frame + [new_label, sub, take_num])

            # Sample transitional frames (label 0)
            unlabeled_indices = np.where(~is_labeled)[0]
            num_transitional = min(2 * n, len(unlabeled_indices))

            if num_transitional > 0:
                sampled_indices = np.random.choice(
                    unlabeled_indices, size=num_transitional, replace=False
                )
                for idx in sampled_indices:
                    flat_frame = mocap_array[idx].flatten().tolist()
                    all_rows.append(flat_frame + [0, sub, take_num])

            mat_file.close()

    if num_joints is None:
        raise RuntimeError("No mocap files were loaded; check your paths/subjects/takes.")

    columns = get_joint_headers(num_joints) + ["Label", "Subject", "Take"]
    output_df = pd.DataFrame(all_rows, columns=columns)
    output_df.to_csv(output_path, index=False)
    print(f"[INFO] Dataset saved to {output_path}")
    return output_path


if __name__ == "__main__":
    # Example direct call (edit values as needed)
    create_dataset_file(
        data_path="Data",
        keypose_csv="assets/keyposes.csv",
        output_dir="Datasets",
        num_takes=2,
        subjects=list(range(1, 11)),
        downsample_rate=5,
        n=20,
        seed=42
    )
