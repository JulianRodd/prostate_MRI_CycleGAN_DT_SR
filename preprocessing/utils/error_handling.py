"""
Error handling utilities for the MRI preprocessing pipeline.
"""

import logging
import sys
import traceback
from functools import wraps
from typing import Any, Callable, Optional, Type, Union


def log_error(identifier: str, message: str) -> None:
    """
    Log an error to the central error log.

    Args:
        identifier: Case ID or error identifier
        message: Error message
    """
    error_logger = logging.getLogger("preprocessing.errors")
    error_logger.error(f"{identifier}: {message}")


def handle_exception(
        func: Optional[Callable] = None,
        reraise: bool = True,
        error_types: Union[Type[Exception], tuple] = Exception,
        logger: Optional[logging.Logger] = None
) -> Callable:
    """
    Decorator to handle exceptions in a standardized way.

    Args:
        func: Function to decorate
        reraise: Whether to reraise the exception after logging
        error_types: Exception types to catch
        logger: Logger to use (defaults to function module's logger)

    Returns:
        Decorated function
    """

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            nonlocal logger

            # Get logger if not provided
            if logger is None:
                logger = logging.getLogger(f.__module__)

            try:
                return f(*args, **kwargs)
            except error_types as e:
                # Extract function name and arguments for context
                func_name = f.__qualname__
                arg_str = ", ".join([str(a) for a in args] + [f"{k}={v}" for k, v in kwargs.items()])

                # Log the error with traceback
                logger.error(f"Error in {func_name}({arg_str}): {str(e)}")
                logger.debug(traceback.format_exc())

                # Log to central error log
                log_error(func_name, str(e))

                # Reraise if required
                if reraise:
                    raise

                # Return None if not reraising
                return None

        return wrapper

    if func is None:
        return decorator
    return decorator(func)

