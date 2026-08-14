"""Model capability and integration errors."""


class ModelCapabilityError(RuntimeError):
    """Raised when an optional model cannot provide a required capability."""


class OptionalDependencyError(ImportError):
    """Raised when an optional integration dependency is unavailable."""
