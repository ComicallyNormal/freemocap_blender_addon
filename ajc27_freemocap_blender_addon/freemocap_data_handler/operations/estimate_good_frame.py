import math
from typing import Dict

import numpy as np


def estimate_good_frame_depr(trajectories_with_error: Dict[str, np.ndarray],
                        velocity_threshold: float = 0.1,
                        ignore_first_n_frames: int = 30):
    trajectory_names = list(trajectories_with_error.keys())
    all_good_frames = None
    all_velocities = {}
    all_errors = {}
    print("trajectory names: ",trajectory_names)
    # if True:
    #     return 0 #TEMP
    # print("trajectories_with_error: ",trajectories_with_error)
    for trajectory_name in trajectory_names:
        # print("sus value: ",trajectories_with_error[trajectory_name]['trajectory']) #looks like it is comparing pose trajectory with hands
        trajectory_velocity_frame_xyz = np.diff(trajectories_with_error[trajectory_name]['trajectory'], axis=0)
        # print("sus value transformed: ",trajectory_velocity_frame_xyz)
        trajectory_velocity_frame_xyz = np.insert(trajectory_velocity_frame_xyz, 0, np.nan, axis=0)
        trajectory_velocity_frame_magnitude = np.sqrt(np.sum(trajectory_velocity_frame_xyz ** 2, axis=1))
        all_velocities[trajectory_name] = trajectory_velocity_frame_magnitude

        trajectory_reprojection_error = trajectories_with_error[trajectory_name]['error']
        all_errors[trajectory_name] = trajectory_reprojection_error

        # define threshold for 'standing still'
        velocity_threshold = np.nanpercentile(trajectory_velocity_frame_magnitude,
                                              q=velocity_threshold * 100)

        # Get the indices of the good frames
        velocity_above_threshold = [index for index, velocity in enumerate(trajectory_velocity_frame_magnitude) if
                                    velocity > velocity_threshold]
        reprojection_error_not_nan = [index for index, error in enumerate(trajectory_reprojection_error) if
                                      not math.isnan(error)]
        floating_point_error = math.ulp(
            1.0) * 10  # 10 times the floating point error, any velocity below this is considered zero
        velocity_not_zero = [index for index, velocity in enumerate(trajectory_velocity_frame_magnitude) if
                             velocity > floating_point_error]

        # To get the good_frames_indices, we need the intersection of the other three lists:
        good_frames_indices = list(
            set(velocity_above_threshold) & set(reprojection_error_not_nan) & set(velocity_not_zero))

        # If no good frames found, skip this trajectory
        if len(good_frames_indices) == 0:
            continue

        # intersect good_frames_indices with all_good_frames
        if all_good_frames is None:
            all_good_frames = set(good_frames_indices)
        else:
            all_good_frames = all_good_frames.intersection(set(good_frames_indices))

    # If no good frames found in any trajectory, raise an Exception
    if  all_good_frames is None:
        raise Exception("No good frames found! Please check your data.")

    # Convert the set to a list
    all_good_frames = list(all_good_frames)

    all_velocities_on_good_frames = np.array([all_velocities[name][all_good_frames] for name in trajectory_names])
    mean_velocities_on_good_frames = np.mean(all_velocities_on_good_frames, axis=0)
    # reprojection_errors_on_good_frames = np.array([all_errors[name][all_good_frames] for name in trajectory_names])

    # normalize the values by their minimum (best) value
    good_frames_velocities_normalized = all_velocities_on_good_frames / np.min(all_velocities_on_good_frames)
    # good_frames_errors_normalized = reprojection_errors_on_good_frames / np.min(reprojection_errors_on_good_frames)

    # Find the index of the good frame with the lowest velocity and lowest error
    combined_error = mean_velocities_on_good_frames# + good_frames_errors_normalized
    best_frame = all_good_frames[np.argmin(combined_error)]

    return best_frame

def estimate_good_frame(trajectories_with_error: Dict[str, np.ndarray],
                        velocity_threshold: float = 20,
                        ignore_first_n_frames: int = 30):
    trajectory_names = list(trajectories_with_error.keys())
    all_good_frames = None
    all_velocities = {}
    all_errors = {}
    print("trajectory names: ",trajectory_names)
    
    for trajectory_name in trajectory_names:
        trajectory_velocity_frame_xyz = np.diff(trajectories_with_error[trajectory_name]['trajectory'], axis=0)
        trajectory_velocity_frame_xyz = np.insert(trajectory_velocity_frame_xyz, 0, np.nan, axis=0)
        trajectory_velocity_frame_magnitude = np.sqrt(np.sum(trajectory_velocity_frame_xyz ** 2, axis=1))
        
        # Clamp negative values to 0 (magnitude should never be negative anyway)
        trajectory_velocity_frame_magnitude = np.maximum(trajectory_velocity_frame_magnitude, 0)
        
        all_velocities[trajectory_name] = trajectory_velocity_frame_magnitude
        trajectory_reprojection_error = trajectories_with_error[trajectory_name]['error']
        all_errors[trajectory_name] = trajectory_reprojection_error
        
        # Check if there are any non-NaN values before calculating percentile
        valid_velocities = trajectory_velocity_frame_magnitude[~np.isnan(trajectory_velocity_frame_magnitude)]
        if len(valid_velocities) == 0:
            print(f"Warning: No valid velocities for {trajectory_name}, skipping...")
            continue
            
        # Define threshold for 'standing still'
        velocity_threshold_value = np.nanpercentile(trajectory_velocity_frame_magnitude,
                                                      q=velocity_threshold)
        
        print("velocity threshold value: ",velocity_threshold)
        # Get the indices of the good frames
        velocity_above_threshold = [index for index, velocity in enumerate(trajectory_velocity_frame_magnitude) if
                                    velocity > velocity_threshold_value]
        reprojection_error_not_nan = [index for index, error in enumerate(trajectory_reprojection_error) if
                                      not math.isnan(error)]
        floating_point_error = math.ulp(1.0) * 10
        velocity_not_zero = [index for index, velocity in enumerate(trajectory_velocity_frame_magnitude) if
                             velocity > floating_point_error]
        
        print("number of items below velocity threshold ",len(velocity_above_threshold))
        print("number of items with valid reprojection error ",len(reprojection_error_not_nan))
        print("number of items with velocity not zero ",len(velocity_not_zero))
        good_frames_indices = list(
            set(velocity_above_threshold) & set(reprojection_error_not_nan) & set(velocity_not_zero))
        
        if len(good_frames_indices) == 0:
            continue
            
        if all_good_frames is None:
            all_good_frames = set(good_frames_indices)
        else:
            all_good_frames = all_good_frames.intersection(set(good_frames_indices))
    
    # Print velocity statistics before the exception check
    print("\n=== Velocity Statistics ===")
    for trajectory_name in trajectory_names:
        if trajectory_name in all_velocities:
            valid_vels = all_velocities[trajectory_name][~np.isnan(all_velocities[trajectory_name])]
            if len(valid_vels) > 0:
                print(f"{trajectory_name}:")
                print(f"  Lowest velocity: {np.min(valid_vels):.6f}")
                print(f"  Highest velocity: {np.max(valid_vels):.6f}")
    print("===========================\n")
    
    if all_good_frames is None or len(all_good_frames) == 0:
        raise Exception("No good frames found! Please check your data.")
    
    all_good_frames = list(all_good_frames)
    all_velocities_on_good_frames = np.array([all_velocities[name][all_good_frames] for name in trajectory_names])
    mean_velocities_on_good_frames = np.mean(all_velocities_on_good_frames, axis=0)
    
    good_frames_velocities_normalized = all_velocities_on_good_frames / np.min(all_velocities_on_good_frames)
    
    combined_error = mean_velocities_on_good_frames
    best_frame = all_good_frames[np.argmin(combined_error)]
    return best_frame