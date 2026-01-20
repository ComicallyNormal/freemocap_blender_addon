from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass
class FreemocapDataPaths:
    body_npy: str
    right_hand_npy: str
    left_hand_npy: str
    face_npy: str
    center_of_mass_npy: str
    segment_centers_of_mass_npy: str
    reprojection_error_npy: str
    calibration_toml: str | None
    hands_enabled : bool = False
    face_enabled :bool = False

    @classmethod
    def from_recording_folder(cls, path: str,hands_enabled : bool, face_enabled :bool):
        print("from_recording_folder entered")
        # print(f"hands enabled? {hands_enabled} face_enabled? {face_enabled} path {path}")
        recording_path = Path(path)
        output_data_path = recording_path / "output_data"

        # TODO: we may want a better form of backwards compatibility than this
        # backwards compatibility:
        center_of_mass_path = Path(output_data_path / "mediapipe_body_total_body_center_of_mass.npy") 
        if not center_of_mass_path.exists():
            center_of_mass_path = Path(output_data_path / "total_body_center_of_mass_xyz.npy")

        print("center of mass path exists? ", center_of_mass_path.exists())
        segment_centers_of_mass_path = output_data_path /  "mediapipe_body_segment_center_of_mass.npy"
        if not segment_centers_of_mass_path.exists():
            segment_centers_of_mass_path = output_data_path / "center_of_mass" / "segmentCOM_frame_joint_xyz.npy"
        
        ignore_error = False
        reprojection_error_path = None
        if not ignore_error:
            reprojection_error_path = output_data_path /"error_manual.npy"
            if not reprojection_error_path.exists():
                print(f"____ERROR: PATH NOT FOUND: {reprojection_error_path}")
                reprojection_error_path = output_data_path / "raw_data" / "mediapipe3dData_numFrames_numTrackedPoints_reprojectionError.npy"
            
        possible_calibration_files = list(recording_path.glob("*calibration.toml"))
        calibration_toml_path = str(possible_calibration_files[0]) if possible_calibration_files else None #for single-cam recording cases where there is no calibration file
        right_hand_path = None
        left_hand_path = None
        face_path = None
        body_path = str(output_data_path / "mediapipe_body_3d_xyz.npy")
        if hands_enabled:
            right_hand_path = str(output_data_path / "mediapipe_right_hand_3d_xyz.npy")
            left_hand_path = str(output_data_path / "mediapipe_left_hand_3d_xyz.npy")
        if face_enabled:
            face_path = str(output_data_path / "mediapipe_face_3d_xyz.npy")

        return cls(
            body_npy=body_path,
            right_hand_npy=right_hand_path,
            left_hand_npy=left_hand_path,
            face_npy=face_path,

            center_of_mass_npy=str(center_of_mass_path),
            segment_centers_of_mass_npy=str(segment_centers_of_mass_path),

            reprojection_error_npy=reprojection_error_path,

            calibration_toml= calibration_toml_path
        )

    @staticmethod
    def _validate_recording_path(recording_path: Union[str, Path]):
        if recording_path == "":
            print("No recording path specified")
            raise FileNotFoundError("No recording path specified")

        if not Path(recording_path).exists():
            print(f"Recording path {recording_path} does not exist")
            raise FileNotFoundError(f"Recording path {recording_path} does not exist")

    def __post_init__(self):
        for path in self.__dict__.values():
            is_bool = type(path) == bool
            if path is None or is_bool:
                continue 
            if not Path(path).exists():
                print(f"Path {path} does not exist")
                raise FileNotFoundError(f"Path {path} does not exist")
