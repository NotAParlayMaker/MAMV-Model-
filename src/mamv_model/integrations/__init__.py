"""Pure payload adapters for MAMV ecosystem boundaries."""

from .mamv import to_mamv_verification_request
from .mamv_ir import to_mamv_ir_workflow_input

__all__ = ["to_mamv_verification_request", "to_mamv_ir_workflow_input"]
