"""Tasks package."""
from .base_task import BaseTask
from .image_pair_tasks import *
from .scene_tasks import *
from .single_image_tasks import *

__all__ = ["BaseTask"]
