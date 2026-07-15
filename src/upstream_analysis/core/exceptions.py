"""Custom exceptions for USP package.

All USP exceptions derive from ``ValueError`` because they ultimately signal an
invalid value (a bad config setting, an invalid data point, or a value that
broke a calculation). This lets callers catch them with ``except ValueError``
while still being able to discriminate via the USP-specific subclasses.
"""


class USPError(ValueError):
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
