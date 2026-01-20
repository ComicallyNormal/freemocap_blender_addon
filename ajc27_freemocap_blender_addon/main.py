import sys
from pathlib import Path

import argparse
import numpy as np


from ajc27_freemocap_blender_addon.core_functions.setup_scene.clear_scene import clear_scene
from ajc27_freemocap_blender_addon.data_models.parameter_models.load_parameters_config import \
    load_default_parameters_config
from ajc27_freemocap_blender_addon.data_models.parameter_models.parameter_models import Config

# from ajc27_freemocap_blender_addon.core_functions.main_controller import MainController


def ajc27_run_as_main_function(recording_path: str,
                               blend_file_path: str,
                               config: Config = load_default_parameters_config()):
    from ajc27_freemocap_blender_addon.core_functions.main_controller import MainController

    controller = MainController(recording_path=recording_path,
                                blend_file_path=blend_file_path,
                                config=config,
                                realtime=False)

    clear_scene()

    controller.load_data()

    print("Done!!!")


def alex_run_as_main_function():
    # from ajc27_freemocap_blender_addon.core_functions.main_controller import MainController
    import ajc27_freemocap_blender_addon.core_functions.main_controller as mc

    print("Loaded main_controller.py from:", mc.__file__)
    inline_points = [{"x":0.0,"y":-0.6,"z":-0.1},{"x":-0.03,"y":-0.65,"z":-0.08},{"x":-0.05,"y":-0.65,"z":-0.08},{"x":-0.07,"y":-0.65,"z":-0.07},{"x":0.03,"y":-0.65,"z":-0.08},{"x":0.05,"y":-0.65,"z":-0.08},{"x":0.07,"y":-0.65,"z":-0.07},{"x":-0.1,"y":-0.63,"z":0.0},{"x":0.1,"y":-0.63,"z":0.0},{"x":-0.02,"y":-0.58,"z":-0.08},{"x":0.02,"y":-0.58,"z":-0.08},{"x":-0.15,"y":-0.45,"z":0.0},{"x":0.15,"y":-0.45,"z":0.0},{"x":-0.2,"y":-0.25,"z":0.0},{"x":0.2,"y":-0.25,"z":0.0},{"x":-0.22,"y":-0.05,"z":0.0},{"x":0.22,"y":-0.05,"z":0.0},{"x":-0.23,"y":0.0,"z":-0.02},{"x":0.23,"y":0.0,"z":-0.02},{"x":-0.24,"y":0.0,"z":0.02},{"x":0.24,"y":0.0,"z":0.02},{"x":-0.21,"y":-0.02,"z":0.03},{"x":0.21,"y":-0.02,"z":0.03},{"x":-0.1,"y":-0.05,"z":0.0},{"x":0.1,"y":-0.05,"z":0.0},{"x":-0.1,"y":0.25,"z":0.0},{"x":0.1,"y":0.25,"z":0.0},{"x":-0.1,"y":0.5,"z":0.0},{"x":0.1,"y":0.5,"z":0.0},{"x":-0.1,"y":0.52,"z":0.05},{"x":0.1,"y":0.52,"z":0.05},{"x":-0.1,"y":0.55,"z":-0.1},{"x":0.1,"y":0.55,"z":-0.1}]
    print(len(inline_points))
    recording_path = "/home/alexmini/Documents/Projects/freemocap_forks/Mediapipe-VR-Fullbody-Tracking/test_data"
    blend_file_path = recording_path
    controller = mc.MainController(recording_path=recording_path,
                                blend_file_path=blend_file_path,
                                config=load_default_parameters_config(),
                                realtime=True)
    
    center_of_mass = np.empty(1) #not real data so don't expect this to run!

    clear_scene()

    reprojection_error = [np.empty(0)]#need json error printout
   
   #once
    controller.load_freemocap_handler_from_data(data_100,reprojection_error,center_of_mass) #Need pose data as ndarray, 
    (center,x_forward,y_left,z_up) = controller.get_ground_data_from_100_frames()

    #continuousely
    moment_of_truth_json = controller.process_mediapipe_pose(inline_points,reprojection_error,center_of_mass,center,x_forward,y_left)
    print("loaded data successfully")
    print(moment_of_truth_json)


def testing_func():
    path = "/home/alexmini/freemocap_data/recording_sessions/session_2026-01-13_22_20_55/recording_22_30_24_gmt-7/output_data/raw_data/mediapipe_3dData_numFrames_numTrackedPoints_reprojectionError.npy"
    error_arr :np.ndarray = np.load(path)
    print(error_arr.shape)
    print(error_arr)

if __name__ == "__main__" or __name__ == "<run_path>":
    print("Loading FREEMOCAP BLENDER ADDON...")
    
    parser = argparse.ArgumentParser(
        description="Run FreeMoCap Blender addon with recording and blender file paths"
    )
    parser.add_argument(
        "--recording",
        type=str,
        required=False,
        help="Path to the recording directory"
    )
    parser.add_argument(
        "--blender",
        type=str,
        required=False,
        help="Path where the blender file should be saved"
    )
    
    try:
        args = parser.parse_args()
        print(f"Received command line arguments: {args}")
        
        # Handle recording path
        if args.recording:
            recording_path_input = Path(args.recording)
        elif __name__ == "<run_path>":
            print("No recording path specified!")
            raise ValueError("No recording path specified")
        elif (Path().home() / "freemocap_data/recording_sessions/freemocap_sample_data").exists():
            recording_path_input = Path().home() / "freemocap_data/recording_sessions/freemocap_sample_data"
        else:
            raise ValueError("No recording path specified")
        
        # Validate recording path exists
        if not recording_path_input.exists():
            print(f"Recording path {recording_path_input} does not exist!")
            raise ValueError(f"Recording path {recording_path_input} does not exist!")
        
        # Handle blender file path
        if args.blender:
            blender_file_save_path_input = Path(args.blender)
        else:
            blender_file_save_path_input = recording_path_input / (recording_path_input.stem + ".blend")
        
        print(f"Running {__file__} with recording_path={recording_path_input}")
        ajc27_run_as_main_function(
            recording_path=str(recording_path_input),
            blend_file_path=str(blender_file_save_path_input)
        )
        # testing_func()
        
    except Exception as e:
        print(f"ERROR RUNNING {__file__}: \n\n GOT ERROR \n\n {str(e)}")