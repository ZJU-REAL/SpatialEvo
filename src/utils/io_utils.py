"""I/O helper functions."""

import json
import yaml
from typing import Dict, Any, Optional
import os


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a JSON or YAML config file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    ext = os.path.splitext(config_path)[1].lower()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if ext in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif ext == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {ext}")


def save_results(
    results: Dict[str, Any],
    output_path: str,
    format: str = "json"
):
    """Save results as JSON or YAML."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        if format == "json":
            json.dump(results, f, indent=2, ensure_ascii=False)
        elif format == "yaml":
            yaml.dump(results, f, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported output format: {format}")
    
    print(f"Saved results to: {output_path}")


def load_scene_data(scene_path: str) -> Dict[str, Any]:
    """Return a placeholder scene data structure."""
    # TODO: Implement actual scene loading.
    
    scene_data = {
        "scene_id": os.path.basename(scene_path),
        "scene_path": scene_path,
        "point_cloud": None,
        "mesh": None,
        "camera_trajectory": [],
        "objects": []
    }
    
    return scene_data


def load_image_list(image_dir: str, extensions: tuple = ('.jpg', '.png', '.jpeg')) -> list:
    """Load all matching images from a directory."""
    if not os.path.exists(image_dir):
        return []
    
    images = []
    for file in sorted(os.listdir(image_dir)):
        if file.lower().endswith(extensions):
            images.append(os.path.join(image_dir, file))
    
    return images


def parse_question_template(
    template: str,
    variables: Dict[str, Any]
) -> str:
    """Fill a question template with variables."""
    question = template
    for key, value in variables.items():
        placeholder = "{" + key + "}"
        question = question.replace(placeholder, str(value))
    
    return question
