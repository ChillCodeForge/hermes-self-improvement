class ImprovementError(Exception):
    """Base exception for expected project errors."""


class AccessDenied(ImprovementError):
    """Raised when a role attempts an operation outside its authority."""


class IntegrityError(ImprovementError):
    """Raised when immutable evaluation evidence no longer matches."""


class PolicyRejected(ImprovementError):
    """Raised when a candidate fails a mandatory policy gate."""
