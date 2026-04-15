import json
import os

def process_data(data):
    processed_data = []
    for it in data:
        scene_id = it['video_id'].split('/')[-1]
        
        frames = []
        for frame_path in it['frame_files']:
            frame_id = frame_path.split('/')[-1].split('.')[0].lstrip('0') or '0'
            frames.append(f'{frame_id}.jpg')
        
        frames.sort(key=lambda x: int(x.split('.')[0]))
        
        processed_data.append({
            'scene_id': scene_id,
            'frame_ids': frames
        })
    
    return processed_data

with open("scannet_select_frames.json", 'r') as f:
    data = json.load(f)

processed_data = process_data(data)
processed_data.sort(key=lambda x: x['scene_id'])

with open("scannet_select_frames_processed.json", 'w') as f:
    json.dump(processed_data, f, indent=4)
    
with open("scannet_scene_to_frames.json", 'w') as f:
    scene_to_frames = {}
    for it in processed_data:
        scene_id = it['scene_id']
        scene_to_frames[scene_id] = it['frame_ids']
    json.dump(scene_to_frames, f, indent=4)