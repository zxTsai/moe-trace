class TraceError(RuntimeError):
    """Base error for trace collection and validation."""


class ConfigError(TraceError):
    """Configuration is invalid or unsupported."""


class IntegrityError(TraceError):
    """Trace data violates the declared schema or completeness rules."""


class TraceWriteError(TraceError):
    """A durable trace write failed."""

