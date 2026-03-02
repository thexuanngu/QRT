class StrategyError(Exception):
    """Raised when a strategy returns invalid values or invalid structures."""


class StrategyNotImplemented(Exception):
    """Raised when a strategy does not implement the required predict method."""
