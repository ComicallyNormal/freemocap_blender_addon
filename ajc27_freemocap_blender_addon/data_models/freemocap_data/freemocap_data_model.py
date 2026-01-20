from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, Literal

import numpy as np
from ajc27_freemocap_blender_addon.core_functions.setup_scene.get_path_to_sample_data import get_path_to_sample_data

from .helpers.freemocap_component_data import FreemocapComponentData
from .helpers.freemocap_data_paths import FreemocapDataPaths
from .helpers.freemocap_data_stats import FreemocapDataStats
from ..mediapipe_names.mediapipe_trajectory_names import MediapipeTrajectoryNames, \
    HumanTrajectoryNames

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None
    try:
        import toml
    except ModuleNotFoundError:
        toml = None


FREEMOCAP_DATA_COMPONENT_TYPES = Literal["body", "right_hand", "left_hand", "face", "other"]


@dataclass
class FreemocapData:
    body: FreemocapComponentData
    hands: Dict[str, FreemocapComponentData]
    face: FreemocapComponentData
    metadata: Optional[Dict[Any, Any]]
    groundplane_calibration: bool = False
    other: Dict[str, FreemocapComponentData] = field(default_factory=dict)
    enable_face : bool = False
    enable_hands : bool = False

    @classmethod
    def from_data(cls,
                  body_frame_name_xyz: np.ndarray,
                  right_hand_frame_name_xyz: np.ndarray,
                  left_hand_frame_name_xyz: np.ndarray,
                  face_frame_name_xyz: np.ndarray,
                  error: np.ndarray,
                  data_source: str = "mediapipe_gpu",
                  error_type: str = "mean_reprojection_error",
                  groundplane_calibration: bool = False,
                  other: Optional[Dict[str, Union[FreemocapComponentData, Dict[str, Any]]]] = None,
                  metadata: Optional[Dict[Any, Any]] = None,
                  enable_face = False,
                  enable_hands = False, #Note, this is called in two places
                  body_data_splice = None
                  ) -> "FreemocapData":

        if not data_source == "mediapipe_gpu":
            raise NotImplementedError(
                f"Data source `{data_source}` not recognized - create the equivalent of `MediapipeTrajectoryNames` for this data source")
        else:
            trajectory_names = MediapipeTrajectoryNames()
        if metadata is None:
            metadata = {}

        if other is None:
            other = {}

        cls._convert_to_component(other=other)

        #NOTE: made error values nullable!!
        (body_error,
         face_error,
         left_hand_error,
         right_hand_error) = cls._split_up_reprojection_error(error=error,
                                                              trajectory_names=trajectory_names,
                                                              enable_face = enable_face,
                                                              enable_hands = enable_hands
                                                              )
        # print("from_data_original error ",error)
        # print("from_data body error: ",body_error)
        calculated_hands = {}
        if enable_hands: 
            calculated_hands = {"right": FreemocapComponentData(name="right_hand",
                                                   data=right_hand_frame_name_xyz,
                                                   data_source=data_source,
                                                   trajectory_names=trajectory_names.right_hand,
                                                   error=right_hand_error,
                                                   error_type=error_type),
                                "left": FreemocapComponentData(name="left_hand",
                                                  data=left_hand_frame_name_xyz,
                                                  data_source=data_source,
                                                  trajectory_names=trajectory_names.left_hand,
                                                  error=left_hand_error,
                                                  error_type=error_type)}
        calculated_face = None
        if enable_face:
            calculated_face = FreemocapComponentData(name="face",
                                        data=face_frame_name_xyz,
                                        data_source=data_source,
                                        trajectory_names=trajectory_names.face,
                                        error=face_error,
                                        error_type=error_type)


        return cls(
            body=FreemocapComponentData(name="body",
                                        data=body_frame_name_xyz,
                                        data_source=data_source,
                                        trajectory_names=trajectory_names.body,
                                        error=body_error,
                                        error_type=error_type),
            
            hands=calculated_hands,
            face=calculated_face,
            other=other,
            groundplane_calibration=groundplane_calibration,
            metadata=metadata,
        )

    @classmethod
    def _convert_to_component(cls, other: Dict[str, Union[FreemocapComponentData, Dict[str, Any]]]):
        for name, component in other.items():
            if isinstance(component, FreemocapComponentData):
                pass
            elif isinstance(component, dict):
                if not "component_name" in component.keys():
                    component["component_name"] = name
                    try:
                        other[name] = FreemocapComponentData(**component)
                    except TypeError as e:
                        print(f"Error creating FreemocapComponentData from dict {component}")
                        raise e
            else:
                raise ValueError(f"Component: {name} type not recognized (type: {type(component)}")
        return other

    @classmethod
    def _split_up_reprojection_error(cls,
                                     error: np.ndarray,
                                     trajectory_names: HumanTrajectoryNames,
                                     enable_face : bool,
                                     enable_hands : bool
                                     ):
        body_names = trajectory_names.body

        right_hand_names = trajectory_names.right_hand
        left_hand_names = trajectory_names.left_hand

        face_names = trajectory_names.face

        body_start = 0
        right_hand_error = None
        left_hand_error = None
        face_error = None

        if(enable_hands):
            right_hand_start = len(body_names)
            left_hand_start = right_hand_start + len(right_hand_names)
            right_hand_error = error[:, right_hand_start:right_hand_start + len(right_hand_names)]
            left_hand_error = error[:, left_hand_start:left_hand_start + len(left_hand_names)]
        
        if(enable_face):
            face_start = left_hand_start + len(left_hand_names) #TODO: hands has a dependency on face

        body_error = error[:, body_start:len(body_names)]

        if(enable_face):
            face_error = error[:, face_start:face_start + len(face_names)]
        
        #TODO: Needs to check if error is none, since frontend v2 isn't giving it.
        # cls._validate_sliced_error(all_error=error,
        #                            trajectory_names=trajectory_names,
        #                            body_error=body_error,
        #                            face_error=face_error,
        #                            left_hand_error=left_hand_error,
        #                            right_hand_error=right_hand_error)

        return body_error, face_error, left_hand_error, right_hand_error

    @classmethod
    def _validate_sliced_error(cls,
                               all_error: np.ndarray,
                               trajectory_names: HumanTrajectoryNames,
                               body_error: np.ndarray,
                               face_error: np.ndarray,
                               left_hand_error: np.ndarray,
                               right_hand_error: np.ndarray):

        body_names = trajectory_names.body
        right_hand_names = trajectory_names.right_hand
        left_hand_names = trajectory_names.left_hand
        face_names = trajectory_names.face

        if not body_error.shape[1] == len(body_names):
            raise ValueError(
                f"Body frame shape {body_error.shape} does not match trajectory names length {len(body_names)}")
        
        #TODO: This doesnt work right when values are non null
        print("type? ",str(type(right_hand_error)))
        if (not right_hand_error == None) and right_hand_error.shape[1] == len(right_hand_names):
            raise ValueError(
                f"Right hand frame shape {right_hand_error.shape} does not match trajectory names length {len(right_hand_names)}")
        if not left_hand_error == None and left_hand_error.shape[1] == len(left_hand_names):
            raise ValueError(
                f"Left hand frame shape {left_hand_error.shape} does not match trajectory names length {len(left_hand_names)}")
        if not face_error == None and face_error.shape[1] == len(face_names):
            raise ValueError(
                f"Face frame shape {face_error.shape} does not match trajectory names length {len(face_names)}")
        all_values_enabled = not right_hand_error == None and not left_hand_error == None and not face_error == None 
        if all_values_enabled and not body_error.shape[1] + right_hand_error.shape[1] + left_hand_error.shape[1] + face_error.shape[1] == \
               all_error.shape[1]:
            raise ValueError(
                f"Error frame shape {all_error.shape} does not match trajectory names length {len(body_names) + len(right_hand_names) + len(left_hand_names) + len(face_names)}")

    @classmethod
    def from_data_paths(cls,
                        data_paths: FreemocapDataPaths,
                        hands_enabled : bool,
                        face_enabled : bool,
                        scale: float = 1000,
                        **kwargs):
        if "metadata" in kwargs.keys():
            metadata = kwargs["metadata"]
        else:
            metadata = {}

        print("from_data_paths entered, metadata was:")
        print(metadata)
        print(f"hands enabled? {hands_enabled} face enabled? {face_enabled}")
        groundplane_calibration:bool = False #set default value
        if data_paths.calibration_toml is not None: #for single-cam recording cases where there is no calibration file and if tomllib is not available
            if tomllib is not None:
                with open (data_paths.calibration_toml, "rb") as f:
                    calibration_data = tomllib.load(f)
                groundplane_calibration = calibration_data.get('metadata', {}).get('groundplane_calibration', False) #for backwards compatibility with files that do not have this key
            elif toml is not None:
                with open (data_paths.calibration_toml, "r", encoding="utf-8") as f:
                    calibration_data = toml.load(f)
                groundplane_calibration = calibration_data.get('metadata', {}).get('groundplane_calibration', False) #for backwards compatibility with files that do not have this key

        # print("loading body data...")
        loaded_body_data : np.ndarray= np.load(data_paths.body_npy) / scale
        # print(f"shape: {loaded_body_data.shape} first pose:")
        converted_body_data = loaded_body_data[:, :33, :] #first 33
        # print(f"new shape: {converted_body_data.shape} first pose:")


        loaded_left_hand_data = None
        loaded_right_hand_data = None
        loaded_face_data = None
        if hands_enabled:
            loaded_left_hand_data = np.load(data_paths.left_hand_npy) / scale
            loaded_right_hand_data = np.load(data_paths.right_hand_npy) / scale
        if face_enabled:
            loaded_face_data = np.load(data_paths.face_npy) / scale


        trajectory_names = MediapipeTrajectoryNames()
        total_error = np.load(data_paths.reprojection_error_npy)
        (body_error,
         face_error,
         left_hand_error,
         right_hand_error) = cls._split_up_reprojection_error(error=total_error,
                                                              trajectory_names=trajectory_names,
                                                              enable_face = False,
                                                              enable_hands = False
                                                              )
        data_source = "mediapipe_gpu"
        error_type = "mean_reprojection_error"
        body_data :FreemocapComponentData =FreemocapComponentData(name="body",
                            data=converted_body_data,
                            data_source=data_source,
                            trajectory_names=trajectory_names.body,
                            error=body_error,
                            error_type=error_type)
        # np.load(data_paths.reprojection_error_npy) TODO: FIXME
        return cls.from_data(
            body_frame_name_xyz=converted_body_data,
            right_hand_frame_name_xyz=loaded_right_hand_data,
            left_hand_frame_name_xyz=loaded_left_hand_data,
            face_frame_name_xyz=loaded_face_data,
            error=total_error,
            
            other={"center_of_mass": FreemocapComponentData(name="center_of_mass",
                                                            data=np.load(
                                                                data_paths.center_of_mass_npy) / scale,
                                                            data_source="freemocap",
                                                            trajectory_names=["center_of_mass"])},
            metadata=metadata,
            groundplane_calibration=groundplane_calibration
        )

    @classmethod
    def from_recording_path(cls,
                            recording_path: str,
                            hands_enabled : bool,
                            face_enabled : bool,
                            **kwargs):
        print("data path initializing")
        data_paths = FreemocapDataPaths.from_recording_folder(path = recording_path,hands_enabled = hands_enabled,face_enabled = face_enabled)
        print("data path returned")
        metadata = {"recording_path": recording_path,
                    "data_paths": data_paths.__dict__}
        print(f"Loading data from paths {data_paths}")
        return cls.from_data_paths(data_paths=data_paths,hands_enabled=hands_enabled,face_enabled=face_enabled, metadata=metadata, **kwargs) #TODO: make configurable

    def __str__(self):
        return str(FreemocapDataStats.from_freemocap_data(self))


if __name__ == "__main__":

    recording_path_in = get_path_to_sample_data()
    freemocap_data = FreemocapData.from_recording_path(recording_path=recording_path_in,
                                                       type="original",
                                                       hands_enabled=False,
                                                       face_enabled=False
                                                       )
    print(str(freemocap_data))
