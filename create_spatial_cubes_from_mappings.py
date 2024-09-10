import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.decomposition import PCA
import numpy.linalg as LA
from scipy.spatial.transform import Rotation as R
import os

with open('/path/to/Detic/all_takes_list.txt') as f:
    all_takes = [x.strip() for x in f.readlines()]

try:
    import sys
    take_name = all_takes[int(sys.argv[1])]
except:
    take_name = "iiith_cooking_45_1"

processed_files = [x for x in os.listdir(f"/path/to/Detic/processed") if f'{take_name}-' in x]
if len(processed_files) == 0:
    print("No processed files available...")
    exit()
dfs = []
for file_path in processed_files:
    df = pd.read_csv(f"/path/to/Detic/processed/{file_path}")
    dfs.append(df)
processed = pd.concat(dfs, ignore_index=True)
processed['object_name'] = processed['object_name'].fillna('no_object')
do_dbscan = True
do_obb = True
print(processed.iloc[0])
print(len(processed))


# 1. Select the top N most frequent object names
# N = 30  # Change this to the desired number of top objects
# top_objects = processed['object_name'].value_counts().head(N).index.tolist()
all_objects = processed['object_name'].value_counts().index.tolist()
# print(f"Top {N} most frequent object names: {top_objects}")
print(len(all_objects))
print(all_objects)


# Filter the DataFrame to only include the top N object names
processed_top =  processed # processed[processed['object_name'].isin(top_objects)].copy()

def uv_to_rgb(row, u_max=640, v_max=480):
    u = row['u']
    v = row['v']
    
    # Normalize u and v to the range [0, 1]
    u_norm = u / u_max
    v_norm = v / v_max
    
    # Calculate RGB values
    r = u_norm  # Red increases with u
    g = v_norm  # Green increases with v
    b = 1 - (u_norm + v_norm) / 2  # Blue decreases as red and green increase
    
    # Clip to the range [0, 1] to avoid invalid colors
    r = np.clip(r, 0, 1)
    g = np.clip(g, 0, 1)
    b = np.clip(b, 0, 1)
    
    return np.array([r, g, b])

def compute_obb_tutorial_xy(df):
    '''
    It takes 3xN as input so first convert the format
    '''
    X = df[['px_world', 'py_world', 'pz_world']].values
    # Filter points between 5th and 95th percentiles along each axis
    lower_percentile = 5
    upper_percentile = 95
    mask = np.ones(len(X), dtype=bool)
    for i in range(3):
        lower = np.percentile(X[:, i], lower_percentile)
        upper = np.percentile(X[:, i], upper_percentile)
        mask &= (X[:, i] >= lower) & (X[:, i] <= upper)
    X_filtered = X[mask]
    data = np.transpose(X_filtered)
    # Only compute means and covariance for x and y
    means_xy = np.mean(data[:2, :], axis=1)
    cov_xy = np.cov(data[:2, :])
    eval_xy, evec_xy = LA.eig(cov_xy)
    centered_data_xy = data[:2, :] - means_xy[:, np.newaxis]
    aligned_coords_xy = np.matmul(evec_xy.T, centered_data_xy)
    # Use original z values
    aligned_coords = np.vstack((aligned_coords_xy, data[2, :]))
    rectCoords = lambda x1, y1, z1, x2, y2, z2: np.array([[x1, x1, x2, x2, x1, x1, x2, x2],
                                                          [y1, y2, y2, y1, y1, y2, y2, y1],
                                                          [z1, z1, z1, z1, z2, z2, z2, z2]])
    realigned_coords_xy = np.matmul(evec_xy, aligned_coords_xy)
    realigned_coords = np.vstack((realigned_coords_xy, data[2, :]))
    realigned_coords[:2, :] += means_xy[:, np.newaxis]
    xmin, xmax = np.min(aligned_coords[0, :]), np.max(aligned_coords[0, :])
    ymin, ymax = np.min(aligned_coords[1, :]), np.max(aligned_coords[1, :])
    zmin, zmax = np.min(aligned_coords[2, :]), np.max(aligned_coords[2, :])
    rrc = np.matmul(evec_xy, rectCoords(xmin, ymin, zmin, xmax, ymax, zmax)[:2, :])
    rrc = np.vstack((rrc, rectCoords(xmin, ymin, zmin, xmax, ymax, zmax)[2, :]))
    rrc[:2, :] += means_xy[:, np.newaxis]
    result = pd.Series({
        'rrc': [rrc],
    })
    return rrc

def compute_obb_tutorial(df):
    '''
    It takes 3xN as input so first convert the format
    '''
    X = df[['px_world', 'py_world', 'pz_world']].values
    # Filter points between 5th and 95th percentiles along each axis
    lower_percentile = 5
    upper_percentile = 95
    mask = np.ones(len(X), dtype=bool)
    for i in range(3):
        lower = np.percentile(X[:, i], lower_percentile)
        upper = np.percentile(X[:, i], upper_percentile)
        mask &= (X[:, i] >= lower) & (X[:, i] <= upper)
    X_filtered = X[mask]
    data = np.transpose(X_filtered)

    means = np.mean(data, axis=1)
    cov = np.cov(data)
    eval, evec = LA.eig(cov)
    eval, evec

    centered_data = data - means[:,np.newaxis]
    # print(np.allclose(LA.inv(evec), evec.T))

    aligned_coords = np.matmul(evec.T, centered_data)

    rectCoords = lambda x1, y1, z1, x2, y2, z2: np.array([[x1, x1, x2, x2, x1, x1, x2, x2],
                                                        [y1, y2, y2, y1, y1, y2, y2, y1],
                                                        [z1, z1, z1, z1, z2, z2, z2, z2]])

    realigned_coords = np.matmul(evec, aligned_coords)
    realigned_coords += means[:, np.newaxis]

    xmin, xmax, ymin, ymax, zmin, zmax = np.min(aligned_coords[0, :]), np.max(aligned_coords[0, :]), np.min(aligned_coords[1, :]), np.max(aligned_coords[1, :]), np.min(aligned_coords[2, :]), np.max(aligned_coords[2, :])

    rrc = np.matmul(evec, rectCoords(xmin, ymin, zmin, xmax, ymax, zmax))

    rrc += means[:, np.newaxis]

    # print(f"Returning shape {rrc.shape}")
    result = pd.Series({
        'rrc': [rrc],
    })

    return rrc

def compute_obb(df):
    # Extract coordinates
    X = df[['px_world', 'py_world', 'pz_world']].values

    # Filter points between 5th and 95th percentiles along each axis
    lower_percentile = 5
    upper_percentile = 95
    mask = np.ones(len(X), dtype=bool)
    for i in range(3):
        lower = np.percentile(X[:, i], lower_percentile)
        upper = np.percentile(X[:, i], upper_percentile)
        mask &= (X[:, i] >= lower) & (X[:, i] <= upper)
    X_filtered = X[mask]

    # Center the data
    mean_point = np.mean(X_filtered, axis=0)
    X_centered = X_filtered - mean_point

    # Perform PCA
    pca = PCA(n_components=3)
    pca.fit(X_centered)

    # Project data onto principal components
    X_pca = X_centered @ pca.components_.T

    # Find min and max along each principal component
    mins = np.min(X_pca, axis=0)
    maxs = np.max(X_pca, axis=0)
    sizes = maxs - mins

    # Center in PCA space
    center_pca = (maxs + mins) / 2

    # Transform center back to original coordinates
    center = center_pca @ pca.components_ + mean_point

    # Create pandas Series with required keys
    result = pd.Series({
        'center_x': center[0],
        'center_y': center[1],
        'center_z': center[2],
        'size_x': sizes[0],
        'size_y': sizes[1],
        'size_z': sizes[2]
    })

    return result

# Function to apply OBB on each cluster
# def apply_oriented_bounding_box(cluster_df):
#     # Extract the point cloud coordinates
#     points = cluster_df[['px_world', 'py_world', 'pz_world']].values
    
#     # Perform PCA to get the principal axes
#     pca = PCA(n_components=3)
#     pca.fit(points)
    
#     # Project the points onto the principal axes
#     transformed_points = pca.transform(points)
    
#     # Find the min and max in the transformed space to get the size of the bounding box
#     min_coords = transformed_points.min(axis=0)
#     max_coords = transformed_points.max(axis=0)
    
#     # Center of the bounding box in transformed space
#     center_transformed = (min_coords + max_coords) / 2
    
#     # Size of the bounding box
#     size = max_coords - min_coords
    
#     # Convert the center back to the original space
#     center_original = pca.inverse_transform(center_transformed)
    
#     return pd.Series({
#         'center_x': center_original[0],
#         'center_y': center_original[1],
#         'center_z': center_original[2],
#         'size_x': size[0],
#         'size_y': size[1],
#         'size_z': size[2],
#     })

# Function to calculate the center and size of the bounding box for a group
def calculate_center_and_size(group):
    x_min, x_max = np.percentile(group['px_world'], [5, 95])
    y_min, y_max = np.percentile(group['py_world'], [5, 95])
    z_min, z_max = np.percentile(group['pz_world'], [5, 95])
    
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    center_z = (z_min + z_max) / 2
    
    size_x = x_max - x_min
    size_y = y_max - y_min
    size_z = z_max - z_min
    
    return pd.Series({
        'center_x': center_x, 'center_y': center_y, 'center_z': center_z,
        'size_x': size_x, 'size_y': size_y, 'size_z': size_z
    })

# Group by 'object_name' and calculate the center and size for each group
if not do_dbscan:
    results = processed_top.groupby('object_name').apply(calculate_center_and_size)

    # Extract the centers, sizes, and object names
    centers = results[['center_x', 'center_y', 'center_z']].to_numpy()
    sizes = results[['size_x', 'size_y', 'size_z']].to_numpy()
    object_names = results.index.tolist()
    export_data = {'centers': centers, 'sizes': sizes, 'object_names': object_names}
    with open(f'{take_name}_object_bbs.pkl', 'wb') as f:
        pickle.dump(export_data, f)
    print(centers, sizes, object_names)

# tv_set = processed_top[processed_top['object_name'] == 'television_set']
# tv_points = processed_top[['px_world', 'py_world', 'pz_world']].to_numpy()
# print(len(processed_top))
# Apply the uv_to_rgb function to each row to generate the color array
# colors = processed_top.apply(uv_to_rgb, axis=1)
# colors = np.stack(colors.to_numpy())
# print(tv_set.head)
# print(tv_points[:5])
# print(tv_set.iloc[0])
# exit()
# assert tv_points.shape == colors.shape
# np.save('all_points_1660_points_debug.npy', tv_points)
# np.save('all_colors_1660_points_debug.npy', colors)

# print(f"Check total points: {len(tv_points)}")

def add_dbscan_clusters_approximate(df, object_name, eps=0.5, min_samples=5):
    # Filter points for the specific object
    df_obj = df[df['object_name'] == object_name].copy()
    
    # Round the coordinates to the nearest 0.1 meters (10 cm)
    df_obj['px_world_rounded'] = df_obj['px_world'].round(1)
    df_obj['py_world_rounded'] = df_obj['py_world'].round(1)
    df_obj['pz_world_rounded'] = df_obj['pz_world'].round(1)
    
    # Remove duplicate rounded points
    df_obj_unique = df_obj.drop_duplicates(subset=['px_world_rounded', 'py_world_rounded', 'pz_world_rounded']).copy()

    print(f"Before dropping duplicates: {len(df_obj)}, after dropping duplicates: {len(df_obj_unique)}")
    
    # Extract the unique rounded coordinates
    points = df_obj_unique[['px_world_rounded', 'py_world_rounded', 'pz_world_rounded']].values
    
    # Run DBSCAN on the unique points
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)
    
    # Add cluster labels to the unique points DataFrame
    df_obj_unique['cluster'] = labels
    df_obj_unique['cluster_label'] = df_obj_unique['cluster'].apply(
        lambda x: f"{object_name}-{x+1}" if x != -1 else f"{object_name}-noise"
    )
    
    # Merge the cluster labels back to the original DataFrame
    df_obj = df_obj.merge(df_obj_unique[['px_world_rounded', 'py_world_rounded', 'pz_world_rounded', 'cluster_label']],
                          on=['px_world_rounded', 'py_world_rounded', 'pz_world_rounded'], how='left')
    
    # Count the number of clusters (excluding noise)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"{object_name} has {num_clusters} clusters (excluding noise)")

    # Drop the temporary rounded columns
    df_obj.drop(columns=['px_world_rounded', 'py_world_rounded', 'pz_world_rounded'], inplace=True)
    
    return df_obj

# 2. Run DBSCAN on each object_name and add a cluster column
def add_dbscan_clusters(df, object_name, eps=0.5, min_samples=5):
    # Filter points for the specific object
    df_obj = df[df['object_name'] == object_name].copy()
    
    # Extract the coordinates
    points = df_obj[['px_world', 'py_world', 'pz_world']].values
    
    # Run DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)
    
    # Add cluster labels
    df_obj['cluster'] = labels
    df_obj['cluster_label'] = df_obj['cluster'].apply(
        lambda x: f"{object_name}-{x+1}" if x != -1 else f"{object_name}-noise"
    )

    # Count the number of clusters (excluding noise)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"{object_name} has {num_clusters} clusters (excluding noise)")
    print(f"Returning back the original dimension of {len(df_obj)}...")
    
    return df_obj

if do_dbscan:
    # Apply DBSCAN and concatenate results
    clustered_dfs = []
    if do_obb:
        export_data = {'rrc': np.empty((0, 3, 8)), 'object_names': [], 'object_counts': []}
    else:
        export_data = {'centers': np.empty((0, 3)), 'sizes': np.empty((0, 3)), 'object_names': [], 'object_counts': []}
    for obj_name in tqdm(all_objects):
        print(f"Clustering object: {obj_name}...")
        clustered_df = add_dbscan_clusters_approximate(processed_top, obj_name, eps=0.5, min_samples=100)
        
        clustered_df = clustered_df[~clustered_df['cluster_label'].str.contains('noise')]
        cluster_counts = clustered_df['cluster_label'].value_counts().to_dict()
        if len(clustered_df) == 0:
            print("All noise, skipping...")
            continue # all noise
        if not do_obb:
            results = clustered_df.groupby('cluster_label').apply(calculate_center_and_size)
            # Extract the centers, sizes, and object names
            centers = results[['center_x', 'center_y', 'center_z']].to_numpy()
            sizes = results[['size_x', 'size_y', 'size_z']].to_numpy()

            export_data['centers'] = np.vstack((export_data['centers'], centers))
            export_data['sizes'] = np.vstack((export_data['sizes'], sizes))

        else:
            results = clustered_df.groupby('cluster_label').apply(compute_obb_tutorial_xy)
            for curr_result_idx in range(len(results)):
                curr_rrc = results.values[curr_result_idx]
                export_data['rrc'] = np.vstack((export_data['rrc'], np.expand_dims(curr_rrc, axis=0)))

        object_names = results.index.tolist()
        curr_object_counts = [cluster_counts[x] for x in object_names]
        export_data['object_names'] = export_data['object_names'] + object_names
        export_data['object_counts'] = export_data['object_counts'] + curr_object_counts
    
    print(f"Final shape: {export_data['rrc'].shape} and final obj length: {len(export_data['object_names'])}")
    with open(f'/path/to/final_object_bbs_with_counts_xy_only/{take_name}_object_bbs_obb.pkl', 'wb') as f:
        pickle.dump(export_data, f)







exit()

# Count the frequency of each object_name
N = 30
frequency = processed['object_name'].value_counts()
top_N_frequency = frequency.head(N)
print(top_N_frequency.head)
# Plot the bar chart
top_N_frequency.plot(kind='bar')

# Customize the plot (optional)
plt.title('Frequency of Object Names')
plt.xlabel('Object Name')
plt.ylabel('Frequency')

# Show the plot
plt.savefig('histogram.png')