"""Custom exceptions for USP package."""


class USPError(Exception):
    """Base exception for USP package."""

    pass


class USPConfigError(USPError):
    """Raised when configuration is invalid or missing required values."""

    pass


class USPDataError(USPError):
    """Raised when data validation fails."""

    pass


class USPCalculationError(USPError):
    """Raised when a calculation fails."""

    pass
