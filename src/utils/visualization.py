"""Visualization helpers."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Optional


def visualize_spatial_relation(
    entity1: Dict[str, Any],
    entity2: Dict[str, Any],
    relation: Dict[str, Any],
    output_path: Optional[str] = None
):
    """Visualize a spatial relation between two entities."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Read positions.
    pos1 = entity1.get("position", [0, 0, 0])
    pos2 = entity2.get("position", [1, 1, 0])
    
    # Draw entities.
    ax.scatter(pos1[0], pos1[1], s=200, c='blue', marker='o', 
               label=entity1.get("name", "Entity 1"))
    ax.scatter(pos2[0], pos2[1], s=200, c='red', marker='s',
               label=entity2.get("name", "Entity 2"))
    
    # Draw the connection line.
    ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'k--', alpha=0.5)
    
    # Add a direction arrow.
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    ax.arrow(pos1[0], pos1[1], dx * 0.8, dy * 0.8,
             head_width=0.1, head_length=0.1, fc='green', ec='green')
    
    # Add a distance label.
    distance = relation.get("distance", 0)
    mid_x = (pos1[0] + pos2[0]) / 2
    mid_y = (pos1[1] + pos2[1]) / 2
    ax.text(mid_x, mid_y, f"Distance: {distance:.2f}m",
            fontsize=12, ha='center')
    
    # Add a direction label.
    direction = relation.get("direction", "Unknown")
    ax.text(pos1[0], pos1[1] - 0.3, f"Direction: {direction}",
            fontsize=10, ha='center')
    
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title("Spatial Relation Visualization")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_execution_timeline(
    step_results: List[Dict[str, Any]],
    output_path: Optional[str] = None
):
    """Plot a rubric execution timeline."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    step_ids = [step["step_id"] for step in step_results]
    tool_names = [step["tool_name"] for step in step_results]
    
    # Draw timeline bars.
    ax.barh(range(len(step_ids)), [1] * len(step_ids), 
            left=range(len(step_ids)), height=0.5)
    
    # Add labels.
    for i, (step_id, tool_name) in enumerate(zip(step_ids, tool_names)):
        ax.text(i + 0.5, i, f"Step {step_id}\n{tool_name}",
                ha='center', va='center', fontsize=9)
    
    ax.set_xlabel("Execution Sequence")
    ax.set_ylabel("Steps")
    ax.set_title("Rubric Execution Timeline")
    ax.set_yticks(range(len(step_ids)))
    ax.set_yticklabels([f"Step {sid}" for sid in step_ids])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved timeline to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_task_distribution(
    execution_history: List[Dict[str, Any]],
    output_path: Optional[str] = None
):
    """Plot task distribution statistics."""
    # Count task types.
    task_counts = {}
    for record in execution_history:
        task_type = record.get("task_type", "unknown")
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    
    # Draw the pie chart.
    fig, ax = plt.subplots(figsize=(10, 8))
    
    labels = list(task_counts.keys())
    sizes = list(task_counts.values())
    
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title("Task Type Distribution")
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved distribution plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()
