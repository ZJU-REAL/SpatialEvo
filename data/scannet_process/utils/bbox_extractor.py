from calendar import c
from math import e
import os
import json
import csv
from joblib import load
import numpy as np
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont
import cv2
import random

INVALID_CLASS = [0, 1, 2, 3, 11, 13, 15, 16, 17, 20, 21, 22, 23, 26, 28, 29, 31, 34, 37, 38, 39, 40, 81]

# Reading key files

def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def read_label_mapping(filename, label_from="raw_category", label_to="nyu40id"):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            mapping[row[label_from]] = row[label_to]
            
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for k, v in mapping.items()}
        
    if represents_int(list(mapping.values())[0]):
        mapping = {k: int(v) for k, v in mapping.items()}
        
    return mapping


def read_mesh_vertices(filename):
    assert os.path.isfile(filename)
    with open(filename, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata["vertex"].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:, 0] = plydata["vertex"].data["x"]
        vertices[:, 1] = plydata["vertex"].data["y"]
        vertices[:, 2] = plydata["vertex"].data["z"]
    return vertices


def read_mesh_vertices_rgb(filename):
    assert os.path.isfile(filename)
    with open(filename, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata["vertex"].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = plydata["vertex"].data["x"]
        vertices[:, 1] = plydata["vertex"].data["y"]
        vertices[:, 2] = plydata["vertex"].data["z"]
        vertices[:, 3] = plydata["vertex"].data["red"]
        vertices[:, 4] = plydata["vertex"].data["green"]
        vertices[:, 5] = plydata["vertex"].data["blue"]
    return vertices

def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data["segGroups"])
        for i in range(num_objects):
            object_id = (
                data["segGroups"][i]["objectId"] + 1
            )  # instance ids should be 1-indexed
            label = data["segGroups"][i]["label"]
            segs = data["segGroups"][i]["segments"]
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data["segIndices"])
        for i in range(num_verts):
            seg_id = data["segIndices"][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts

def read_scene_axis_alignment(meta_file):
    lines = open(meta_file).readlines()
    for line in lines:
        if "axisAlignment" in line:
            axis_align_matrix = [
                float(x) for x in line.rstrip().strip("axisAlignment = ").split(" ")
            ]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    
    return axis_align_matrix

def align_mesh_vertices(mesh_vertices, axis_align_matrix):
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:, 0:3] = mesh_vertices[:, 0:3]
    pts = np.dot(pts, axis_align_matrix.transpose())
    mesh_vertices[:, 0:3] = pts[:, 0:3]
    
    return mesh_vertices

def get_object_id_to_label_id(agg_file, seg_file, raw_id_map):
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    object_id_to_label_id = {}
    for label, segs in label_to_segs.items():
        label_id = raw_id_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    num_instances = len(np.unique(list(object_id_to_segs.keys())))
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = object_id
            if object_id not in object_id_to_label_id:
                object_id_to_label_id[object_id] = label_ids[verts][0]

    return object_id_to_label_id, object_id_to_segs, instance_ids


# Exporting 3D BBox
def draw(image, points, frame_id):
    if type(image) == str:
        image = Image.open(image)
    
    draw = ImageDraw.Draw(image)
    for point in points:
        px, py = int(point[0]), int(point[1])
        draw.ellipse([px-2, py-2, px+2, py+2], fill='red')
        
    save_dir = "first_frame_visualization"
    os.makedirs(save_dir, exist_ok=True)
    
    image.save(f"first_frame_visualization/{frame_id}.jpg")


def get_first_frame_id(axis_align_matrix, points, colors, scene_dir):
    global id
    for frame in colors:
        color_file = os.path.join(scene_dir, 'color', frame)
        depth_file = os.path.join(scene_dir, 'depth', frame.replace('.jpg', '.png'))
        pose_file = os.path.join(scene_dir, 'pose', frame.replace('.jpg', '.txt'))
        camera_intrinsic_file = os.path.join(scene_dir, 'intrinsic_color.txt')
        
        intrinsic = load_matrix_from_txt(camera_intrinsic_file)
        pose = load_matrix_from_txt(pose_file)
        
        color = Image.open(color_file)
        shape = color.size
        depth = cv2.imdecode(np.fromfile(depth_file, dtype=np.uint8), -1)
        
        pose = axis_align_matrix @ pose
        
        visible_points, visibility, _, _, _ = project_points(points, intrinsic, pose, shape, depth, occlusion_threshold=0.001)
        if visibility >= 0.1:
            # draw(color, visible_points, frame.replace('.jpg', ''))
            return frame
    return "-1"

def export_3d_bbox(axis_align_matrix, aligned_mesh_vertices, object_id_to_label_id, object_id_to_segs, instance_ids, id_label_map, scene_dir, visualize=False):
    colors = os.listdir(os.path.join(scene_dir, 'color'))
    colors.sort(key = lambda x: int(x.replace('.jpg', '')))
    colors = colors[::20]
    
    instance_bboxes = []
    for obj_id in object_id_to_segs:
        label_id = object_id_to_label_id[obj_id]
        
        if label_id in INVALID_CLASS:
            continue
        
        obj_pc = aligned_mesh_vertices[instance_ids == obj_id, 0:3]
        if len(obj_pc) == 0:
            continue
        # Compute axis aligned box
        # An axis aligned bounding box is parameterized by
        # (cx,cy,cz) and (dx,dy,dz) and label id
        # where (cx,cy,cz) is the center point of the box,
        # dx is the x-axis length of the box.
        xmin = np.min(obj_pc[:, 0])
        ymin = np.min(obj_pc[:, 1])
        zmin = np.min(obj_pc[:, 2])
        xmax = np.max(obj_pc[:, 0])
        ymax = np.max(obj_pc[:, 1])
        zmax = np.max(obj_pc[:, 2])
        first_frame_id = get_first_frame_id(axis_align_matrix, obj_pc, colors, scene_dir)
        bbox = np.array(
            [
                xmin,
                ymin,
                zmin,
                xmax,
                ymax,
                zmax,
                id_label_map[label_id],
                first_frame_id,
                obj_id,
            ]
        )
        instance_bboxes.append(bbox)
        
        if visualize:
            bboxes_3d_visualization(instance_bboxes)
    
    return instance_bboxes

def bboxes_3d_visualization(input, out_path='bboxes_3d_visualization.jpg'):
    bboxes = []
    labels = []
    obj_ids = []
    
    for i, item in enumerate(input):
        box_data = [float(val) for val in item[:6]]
        label = item[6] 
        obj_id = item[-1]
        
        bboxes.append(box_data)
        labels.append(label)
        obj_ids.append(obj_id)

    label_set = list(set(labels))
    label_to_color = {}
    colors = list(mcolors.TABLEAU_COLORS.values())
    for i, label in enumerate(label_set):
        label_to_color[label] = colors[i % len(colors)]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i, (bbox, label, obj_id) in enumerate(zip(bboxes, labels, obj_ids)):
        x_min, y_min, z_min, x_max, y_max, z_max = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]
        
        vertices = [
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max]
        ]
        
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[3], vertices[7], vertices[4]]
        ]
        
        color = label_to_color[label]
        collection = Poly3DCollection(faces, alpha=0.25, facecolor=color, edgecolor='black', linewidth=1)
        ax.add_collection3d(collection)
        
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        center_z = (z_min + z_max) / 2
        ax.text(center_x, center_y, center_z, f"{obj_id}:{label}", color='black', fontsize=8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    bounds = []
    for bbox in bboxes:
        x, y, z = bbox[0], bbox[1], bbox[2]
        w, h, d = bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]
        bounds.extend([(x, y, z), (x+w, y+h, z+d)])
    bounds = np.array(bounds)
    
    if(len(bounds)) == 0:
        x_min, y_min, z_min = -2, -2, -2
        x_max, y_max, z_max = 2, 2, 2
    else:
        x_min, y_min, z_min = np.min(bounds, axis=0)
        x_max, y_max, z_max = np.max(bounds, axis=0)

    margin = 0.5
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_zlim(z_min - margin, z_max + margin)

    handles = [plt.Rectangle((0, 0), 1, 1, color=label_to_color[label]) for label in label_set]
    ax.legend(handles, label_set, loc='upper right', bbox_to_anchor=(1.1, 1))

    ax.view_init(elev=90, azim=-90)

    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    plt.close()


# Exporting 2D BBox

def load_matrix_from_txt(path, shape=(4, 4)):
    with open(path) as f:
        txt = f.readlines()
    txt = ''.join(txt).replace('\n', ' ')
    matrix = [float(v) for v in txt.split()]
    return np.array(matrix).reshape(shape)

def project_points(points, intrinsic, pose, shape, depth=None, depth_scale=1000, occlusion_threshold=0.001):
    depth_height, depth_width = 0, 0
    depth_to_color_scale_x, depth_to_color_scale_y = 1.0, 1.0
    
    truncated, occluded, usable = False, False, True
    
    if depth is not None:
        depth_height, depth_width = depth.shape[:2]
        depth_to_color_scale_x = shape[0] / depth_width
        depth_to_color_scale_y = shape[1] / depth_height
        
    world_to_cam = np.linalg.inv(pose)
    
    world_points = np.array([[point[0], point[1], point[2], 1] for point in points])
    cam_points = world_points @ world_to_cam.T
    
    visibility = cam_points[:, 2] > 0
    
    points_image = cam_points @ intrinsic.T
    
    points_image = points_image[:, :2] / points_image[:, 2:3]
    
    in_image = (points_image[:, 0] >= 0) & (points_image[:, 0] < shape[0]) & \
               (points_image[:, 1] >= 0) & (points_image[:, 1] < shape[1])
    
    if sum(in_image) != 0:
        truncated = True
    
    visibility &= in_image
    
    if depth is not None:
        for i in range(len(points_image)):
            if visibility[i]:
                color_x, color_y = int(points_image[i, 0]), int(points_image[i, 1])
                depth_x, depth_y = int(color_x / depth_to_color_scale_x), int(color_y / depth_to_color_scale_y)
                
                if 0 <= depth_x < depth_width and 0 <= depth_y < depth_height:
                    actual_depth = float(depth[depth_y, depth_x]) / depth_scale
                    
                    calculated_depth = float(cam_points[i, 2])
                    
                    eps = 1e-6
                    relative_error = (calculated_depth - actual_depth) / (actual_depth + eps)
                    if actual_depth > 0 and relative_error > occlusion_threshold:
                        visibility[i] = False
                        occluded = True
            
            else:
                pass
    
    visible_points = points_image[visibility]
    visibility_ratio = len(visible_points) / len(points_image)
    
    # Calculate center camera coordinates
    if sum(visibility) != 0:
        visible_cam_points = cam_points[visibility][:, :3]
        minc = np.min(visible_cam_points, axis=0)
        maxc = np.max(visible_cam_points, axis=0) 
        cam_loc = (minc + maxc) / 2 
    else:
        cam_loc = np.array([-1, -1, -1])
    
    
    return visible_points, round(visibility_ratio, 4), truncated, occluded, cam_loc
        
        

def export_2d_bbox(axis_align_matrix, aligned_mesh_vertices, object_id_to_label_id, object_id_to_segs, instance_ids, camera_intrinsic_file, color_file, depth_file, pose_file, id_label_map, visualize=False):
    intrinsic = load_matrix_from_txt(camera_intrinsic_file)
    pose = load_matrix_from_txt(pose_file)
    
    color = Image.open(color_file)
    shape = color.size
    depth = cv2.imdecode(np.fromfile(depth_file, dtype=np.uint8), -1)
    
    pose = axis_align_matrix @ pose
    
    # Get the object points
    instance_bboxes = []
    for obj_id in object_id_to_segs:
        label_id = object_id_to_label_id[obj_id]
        
        if label_id in INVALID_CLASS:
            continue
        
        obj_pc = aligned_mesh_vertices[instance_ids == obj_id, 0:3]
        if len(obj_pc) == 0:
            continue
        
        points_image, visibility, truncated, occluded, cam_loc = project_points(obj_pc, intrinsic, pose, shape, depth=depth, occlusion_threshold=0.001)
        if len(points_image) == 0:
            continue
        
        # Compute axis aligned box
        xmin = np.min(points_image[:, 0])
        ymin = np.min(points_image[:, 1])
        xmax = np.max(points_image[:, 0])
        ymax = np.max(points_image[:, 1])
        
        margin = 2
        
        bbox = [
                min(shape[0], max(0, int(xmin) - margin)),
                min(shape[1], max(0, int(ymin) - margin)),
                min(shape[0], int(xmax + 0.5) + margin),
                min(shape[1], int(ymax + 0.5) + margin),
                points_image,
                id_label_map[label_id],
                visibility,
                cam_loc,
                truncated,
                occluded,
                obj_id,
               ]
        
        instance_bboxes.append(bbox)
        
        # if visualize:
        #     bboxes_2d_visualization(instance_bboxes, color)    
        
    return instance_bboxes
    
def bboxes_2d_visualization(input, image, visibility_threshold):
    if type(image) == str:
        image = Image.open(image)
    
    draw = ImageDraw.Draw(image)
    
    for bbox in input:
        visibility = float(bbox[6])
        
        if visibility < visibility_threshold:
            continue
        
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        points = bbox[4]
        label = bbox[5]
        obj_id = bbox[-1]
        
        
        colors = ["#FF5733", "#33FF57", "#3357FF", "#FF33A6", "#33FFF5", "#F5FF33", "#A633FF", "#FF8C33", "#8CFF33", "#337BFF", "#FF33F5", "#33FFB1", "#FFD633", "#7D33FF", "#FF3352"]
        color = random.choice(colors)
        
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        text = str(obj_id) + ': ' + str(label) + ' visibility: ' + str(visibility)
        
        text_bbox = draw.textbbox((x1+2, y1+2), text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        draw.rectangle([(x1+2, y1+2), (x1+2+text_width, y1+2+text_height)], fill="white")
        
        draw.text((x1+2, y1+2), text, fill="black")
        
        
        # for point in points:
        #     px, py = int(point[0]), int(point[1])
        #     draw.ellipse([px-2, py-2, px+2, py+2], fill=color)
    
    # image.save("bbox_2d_visualization.jpg")
    return image
    
        
