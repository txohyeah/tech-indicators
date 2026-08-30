"""tech-indicators 包内异常定义。

从 stock-research 的 exceptions.py 迁移，基类更名为 TechIndicatorsError。
保留 error_code / exit_code / hint 属性，供上层 CLI 映射退出码。
"""


class TechIndicatorsError(Exception):
    """Base exception with an agent-readable error code and exit status."""

    error_code = "INTERNAL_ERROR"
    exit_code = 5
    hint = ""

    def __init__(self, message: str, *, hint: str | None = None, payload: dict[str, object] | None = None) -> None:
        super().__init__(message)
        if hint is not None:
            self.hint = hint
        self.payload = payload or {}


class UserInputError(TechIndicatorsError):
    error_code = "INVALID_ARGUMENT"
    exit_code = 1


class DatabaseConnectionError(TechIndicatorsError):
    error_code = "DATABASE_CONNECTION_FAILED"
    exit_code = 2
    hint = "Check --env or MYSQL_* settings"


class DataInsufficientError(TechIndicatorsError):
    error_code = "DATA_INSUFFICIENT"
    exit_code = 3


class ReportWriteError(TechIndicatorsError):
    error_code = "REPORT_WRITE_FAILED"
    exit_code = 4