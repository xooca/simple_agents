# agentic_orchestrator/utils/api_key_utils.py
import os

def get_api_key(env_var_name: str, service_name: str, required: bool = True) -> str | None:
    """
    Retrieves an API key from environment variables.

    Args:
        env_var_name: The name of the environment variable.
        service_name: The name of the service for error/warning messages.
        required: If True, raises an error if the key is not found.
                  If False, prints a warning and returns None.

    Returns:
        The API key string, or None if not found and not required.

    Raises:
        ValueError: If the API key is required and not found.
    """
    api_key = os.getenv(env_var_name)
    if not api_key:
        if required:
            raise ValueError(f"{service_name} API key not found. Please set the {env_var_name} environment variable.")
        else:
            print(f"Warning: {service_name} API key not set via {env_var_name}. Certain functionalities may not work.")
    return api_key