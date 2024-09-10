'''
This code tries to use PC and combine that with detic predictions to create a bounding box for every object.

Steps:
1. Load the takes.json and timesync.csv to find the range of acceptable times in us/ns
2. Choose only those points in observations.csv.gz that have the time in this acceptable range
3. For the chosen points
'''

'''
CRITICAL ASSUMPTIONS TO CHECK
-----------------------------
camera-slam-left -- does it always correspond to the 1 and right corresponds to 2? YES
how is u, v mapped to image dimension? I am assuming u is the 1st dimension since u goes as high as 640 and frame shape is 480, 640? CORRECTED
'''

import pandas as pd
import swifter
import os
import json
import cv2
import pickle
from tqdm import tqdm

import sys
if True:
    take_name = sys.argv[1]
    start_frame = int(sys.argv[2])
    end_frame = int(sys.argv[3])
else:
    take_name = 'iiith_cooking_45_1'
save_location_name = f"{take_name}-{start_frame}-{end_frame}"
low_fps = True

basedir = '/datasets01/egoexo4d/v2/takes/'
detic_output_path = '/path/to/Detic/predictions/'
# Load the vocabulary
with open('/path/to/Detic/datasets/metadata/lvis_v1_train_cat_info.json') as f:
    vocabulary = json.load(f)

# Load the instances dict for faster processing
camera_name_to_output_mapping = {
    'camera-slam-left': 'aria02_1201-1',
    'camera-slam-right': 'aria02_1201-2',
}
instances_dict = {}
for key in ['camera-slam-left', 'camera-slam-right']:
    all_instance_paths = os.listdir(os.path.join(detic_output_path, save_location_name, key))
    instances_dict[key] = {}
    for idx in tqdm(range(len(all_instance_paths))):
        with open(os.path.join(detic_output_path, save_location_name, key, all_instance_paths[idx]), 'rb') as f:
            instances_dict[key][all_instance_paths[idx]] = pickle.load(f)
print("Done populating instances for later use...")

def find_object_name_and_index(row):
    # camera_name_to_output_mapping = {
    #     'camera-slam-left': 'output_predictions_1',
    #     'camera-slam-right': 'output_predictions_2',
    # }
    # instance_path = os.path.join(detic_output_path, take_name, camera_name_to_output_mapping[row['camera_name']], f"{row['frame_idx']:05}.pkl")
    # with open(instance_path, 'rb') as f:
    #     instances = pickle.load(f)
    assert row['u'] >= 0.0 and row['u'] <= 640.0 and row['v'] >= 0.0 and row['v'] <= 480.0
    instances = instances_dict[row['camera_name']][f"{row['frame_idx']:06}.pkl"]
    pred_masks = instances.pred_masks
    pred_classes = instances.pred_classes
    # print(pred_masks.shape, "*"*100)
    assert pred_masks.shape[1:3] == (640, 480) or pred_masks.shape[1:3] == (480, 640)
    if pred_masks.shape[1:3] == (640, 480):
        # Same shape as the depth map, so everything is good
        curr_pred_masks = pred_masks[:, int(row['u']), 480 - int(row['v'])]
    else:
        # Apply depth value to mask correction
        if int(row['u']) < 80 or int(row['u']) > 559:
            object_index = -100
            object_name = "NA"
            return object_name, object_index
        # For these samples we do have the segmentation mask
        curr_pred_masks = pred_masks[:, int(row['u']) - 80, 80 + 480 - int(row['v'])]
    assert pred_classes.shape == curr_pred_masks.shape
    # Find the first index where bool_values is True
    if curr_pred_masks.any():
        object_index = pred_classes[curr_pred_masks].min().item()
        object_name = vocabulary[object_index]['name']
        return object_name, object_index
    else:
        object_index = -100
        object_name = "NA"
        return object_name, object_index


# Load the observation file and online_calibration to find the mapping between camera and the camera ID
observations = pd.read_csv(os.path.join(basedir, take_name, 'trajectory/semidense_observations.csv.gz'))
points = pd.read_csv(os.path.join(basedir, take_name, 'trajectory/semidense_points.csv.gz'))
online_calibrations = [json.loads(line) for line in open(os.path.join(basedir, take_name, 'trajectory/online_calibration.jsonl'), 'r')]

# Check if the number of rows in online_calibration is roughly same as the number of frames in the SLAM videos.
total_frames = len(os.listdir(os.path.join(detic_output_path, save_location_name, 'camera-slam-left')))
# if os.path.exists(os.path.join(basedir, take_name, 'frame_aligned_videos/aria01_1201-1.mp4')):
#     video = cv2.VideoCapture(os.path.join(basedir, take_name, 'frame_aligned_videos/aria01_1201-1.mp4'))
# else:
#     video = cv2.VideoCapture(os.path.join(basedir, take_name, 'frame_aligned_videos/aria02_1201-1.mp4'))
# total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
# assert total_frames == len(online_calibrations), f"Inconsistent frame count: {total_frames} vs {len(online_calibrations)}"

print(f"Original observations size: {len(observations)}")
# Filter to keep only correct timestamp ids
observations = observations[observations['frame_tracking_timestamp_us'].isin([x['tracking_timestamp_us'] for x in online_calibrations])]
print(f"After removing observations without a timestamp: {len(observations)}")
# Add a field that names the camera
serial_to_camera_name = {online_calibrations[0]['CameraCalibrations'][x]['SerialNumber']: online_calibrations[0]['CameraCalibrations'][x]['Label'] for x in range(len(online_calibrations[0]['CameraCalibrations']))}

# Double check the serial number and num cameras remain constant
assert all(len(online_calibrations[x]['CameraCalibrations']) == len(online_calibrations[0]['CameraCalibrations']) for x in range(len(online_calibrations)))
for camera_idx in range(len(online_calibrations[0]['CameraCalibrations'])):
    assert all((online_calibrations[x]['CameraCalibrations'][camera_idx]['SerialNumber'] == online_calibrations[0]['CameraCalibrations'][camera_idx]['SerialNumber'] and online_calibrations[x]['CameraCalibrations'][camera_idx]['Label'] == online_calibrations[0]['CameraCalibrations'][camera_idx]['Label']) for x in range(len(online_calibrations)))
assert all(serial_to_camera_name[x] in ['camera-slam-left', 'camera-slam-right', 'camera-rgb'] for x in serial_to_camera_name)
assert all(x in list(serial_to_camera_name.keys()) for x in observations['camera_serial'].unique().tolist())

# Add camera name as a column
observations['camera_name'] = observations['camera_serial'].map(serial_to_camera_name)
# Add frame idx
observations['frame_idx'] = observations['frame_tracking_timestamp_us'].map({online_calibrations[x]['tracking_timestamp_us']: x for x in range(len(online_calibrations))})
# Merge observations and points
if not set(observations['uid']).issubset(set(points['uid'])):
    missing_uids = set(observations['uid']) - set(points['uid'])
    raise KeyError(f"The following uid(s) from observations are not found in points: {missing_uids}")
observations = pd.merge(observations, points[['uid', 'px_world', 'py_world', 'pz_world', 'inv_dist_std', 'dist_std']], on='uid', how='inner')

# Finally, use object segmentation and add object
print("Finding object and instances...")
observations = observations[(observations['frame_idx'] >= start_frame) & (observations['frame_idx'] < end_frame)]
if low_fps:
    observations = observations[observations['frame_idx'] % 6 == 0]
print(f"Length of the final array being processed: {len(observations)}")

tqdm.pandas()  # Initialize tqdm with pandas
observations[['object_name', 'object_index']] = observations.swifter.apply(lambda row: find_object_name_and_index(row), axis=1, result_type='expand')
print("Dropping 3d points without an object name...")
observations = observations[observations['object_index'] != -100]
print(f"Final observations size with at least one object...: {len(observations)}")
os.makedirs("/path/to/temp_data/Detic/processed/", exist_ok=True)
observations.to_csv(f'/path/to/temp_data/Detic/processed/{save_location_name}_processed.csv.gz', index=False, compression='gzip')
print(observations.iloc[0])
exit()

print(len(online_calibrations))
print(observations.head)
print(online_calibrations[0].keys())
for key in online_calibrations[0]:
    print(f"Keys: {key}, Value: {online_calibrations[0][key]}")


