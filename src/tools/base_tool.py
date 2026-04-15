"""Base tool definitions."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """Base class for all tools."""

    def __init__(self, name: str, description: str, aliases: tuple[str, ...] = ()):
        """Initialize the tool."""
        self.name = name
        self.description = description
        self.aliases = tuple(aliases)

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool."""
        pass

    def validate_inputs(self, required_keys: list, kwargs: dict) -> bool:
        """Check whether all required inputs are present."""
        return all(key in kwargs for key in required_keys)

    def get_info(self) -> Dict[str, Any]:
        """Return tool metadata."""
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "description": self.description
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
