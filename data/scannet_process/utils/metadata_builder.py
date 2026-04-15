import numpy as np


def export_scene_metadata(scene_id, bboxes_3d, room_size):
    scene_metadata = {
        'scene_id': scene_id,
        'room_size': room_size,
        'objects': []
    }
    
    for bbox in bboxes_3d:
        obj_id = int(bbox[-1])
        first_frame = bbox[-2]
        label = bbox[-3]
        bbox_3d = list(map(float, bbox[0:6]))
        location_3d = [(bbox_3d[i] + bbox_3d[i + 3]) / 2 for i in range(3)]
        size = max([bbox_3d[i + 3] - bbox_3d[i] for i in range(3)]) * 100
        
        scene_metadata['objects'].append({
            'object_id': obj_id,
            'label': label,
            'size': size,
            '3d_bbox': bbox_3d,
            '3d_location': location_3d,
            'first_frame': first_frame
        })
    
    return scene_metadata

def export_frame_metadata(scene_id, frame_id, bboxes_2d, obj_id_to_3d_bbox, obj_id_to_3d_loc, obj_id_to_size):
    frame_metadata = {
        'scene_id': scene_id,
        'frame_id': frame_id,
        'objects': []
    }
    
    for bbox in bboxes_2d:
        obj_id = int(bbox[-1])
        label = bbox[5]
        visibility = float(bbox[6])
        cam_loc = list(bbox[7])
        truncated = bbox[8]
        occluded = bbox[9]
        bbox_2d = list(map(int, bbox[0:4]))
        bbox_3d = obj_id_to_3d_bbox[obj_id]
        loc_3d = obj_id_to_3d_loc[obj_id]
        size = obj_id_to_size[obj_id]
        
        frame_metadata['objects'].append({
            'object_id': obj_id,
            'label': label,
            'size': size,
            'visibility': visibility,
            'truncated': truncated,
            'occluded': occluded,
            '2d_bbox': bbox_2d,
            'camera_location': cam_loc,
            '3d_bbox': bbox_3d,
            '3d_location': loc_3d
        })
        
    return frame_metadata

def frame_metadata_to_bbox_2d(frame_metadata):
    bboxes = []
    for obj_metadata in frame_metadata['objects']:
        obj_id = int(obj_metadata['object_id'])
        (xmin, ymin, xmax, ymax) = (int(it) for it in obj_metadata['2d_bbox'])
        visibility = float(obj_metadata['visibility'])
        truncated = obj_metadata['truncated']
        occluded = obj_metadata['occluded']
        label = obj_metadata['label']
        cam_loc = np.array(obj_metadata['camera_location'])
        
        bboxes.append(
            [
                xmin,
                ymin,
                xmax,
                ymax,
                [],
                label,
                visibility,
                truncated,
                cam_loc,
                occluded,
                obj_id
            ]
        )
    
    return bboxes

def scene_metadata_to_bbox_3d(scene_metadata):
    bboxes = []
    for obj_metadata in scene_metadata['objects']:
        obj_id = int(obj_metadata['object_id'])
        (xmin, ymin, zmin, xmax, ymax, zmax) = (float(it) for it in obj_metadata['3d_bbox'])
        label = obj_metadata['label']
        first_frame_id = obj_metadata['first_frame']
        
        bboxes.append(
            [
                xmin,
                ymin,
                zmin,
                xmax,
                ymax,
                zmax,
                label,
                first_frame_id,
                obj_id,
            ]
        ) 
    
    return bboxes
        