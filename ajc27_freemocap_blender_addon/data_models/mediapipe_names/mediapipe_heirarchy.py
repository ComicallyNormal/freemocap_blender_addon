# Dictionary containing the empty children for each of the capture empties.
# This will be used to correct the position of the empties (and its children) that are outside the bone length interval defined by x*stdev
from copy import deepcopy


# leading underscore to denote that this is private and should not be used/accessed directly. Use the `getter` function below
_MEDIAPIPE_HIERARCHY = {
    # BODY
    # TORSO
    'hips_center': {
        'children': ['right_hip',
                     'left_hip',
                     'trunk_center']
    },
    'trunk_center': {
        'children': ['neck_center']
    },
    'neck_center': {
        'children': ['right_shoulder',
                     'left_shoulder',
                     'head_center']
    },
    'head_center': {
        'children': [
            'nose',
                     'mouth_right',
                     'mouth_left',
                     'right_eye',
                     'right_eye_inner',
                     'right_eye_outer',
                     'left_eye',
                     'left_eye_inner',
                     'left_eye_outer',
                     'right_ear',
                     'left_ear'
                     ]
    },
    # LEGS
    # RIGHT LEG
    'right_hip': {
        'children': ['right_knee']
    },
    'right_knee': {
        'children': ['right_ankle']
    },
    'right_ankle': {
        'children': ['right_foot_index',
                     'right_heel']
    },
    # LEFT LEG
    'left_hip': {
        'children': ['left_knee']
    },

    'left_knee': {
        'children': ['left_ankle']
    },

    'left_ankle': {
        'children': ['left_foot_index',
                     'left_heel']},

    # ARMS
    # RIGHT ARM
    'right_shoulder': {
        'children': ['right_elbow']
    },
    'right_elbow': {
        'children': ['right_wrist']
    },
    # LEFT ARM
    'left_shoulder': {
        'children': ['left_elbow']
    },

    'left_elbow': {
        'children': ['left_wrist']
    },
    # HANDS
    # RIGHT HAND
    
    # LEFT HAND
}

def get_mediapipe_hierarchy():
    """
    Return a copy of the mediapipe hierarchy, to ensure the base definitions isn't modified.
    """
    return deepcopy(_MEDIAPIPE_HIERARCHY)