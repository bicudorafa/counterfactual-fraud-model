"""Backward compatibility adapters for existing API."""

from .adapters import (
    LegacyDataGenerator,
    LegacySyntheticDataGenerator
)

__all__ = [
    "LegacyDataGenerator",
    "LegacySyntheticDataGenerator"
] 