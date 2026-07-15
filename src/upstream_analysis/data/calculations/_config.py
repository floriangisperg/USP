"""Shared configuration helpers for the fermentation-data calculation modules.

These were promoted out of the (now thin) :class:`FermentationDataProcessor` so
that each focused calculation submodule can read the merged config exactly the
way the monolithic class used to. Logic is preserved verbatim; only the
``self.config`` / ``self.processing_config`` bindings became explicit arguments.
"""

import logging
from typing import List, Optional

from ...core.exceptions import USPConfigError

logger = logging.getLogger(__name__)

# Config-key name constants (were ``self.*_key`` on the old class).
PROCESSING_PARAMETERS_KEY = "processing_parameters"
FEED_DEFINITIONS_KEY = "feed_definitions"
BASE_TITRANT_KEY = "base_titrant"
REACTOR_SETUP_KEY = "reactor_setup"

# Parameter-name substrings whose numeric config values must be strictly
# positive. Note ``initial_volume_ml`` is intentionally NOT here; its
# positivity is enforced separately by ``FermentationDataProcessor._validate_config``.
_POSITIVE_PARAM_HINTS = (
    "density_g_ml",
    "molarity_mol_l",
    "start_volume_ml",
    "molar_volume_air_l_mol",
)


def get_processing_config(config: dict) -> dict:
    """Resolve the ``processing_parameters`` section, supporting old and new layouts.

    Resolution order (unchanged from the original method):
    ``config['processing']['processing_parameters']`` -> ``config['processing']``
    -> ``config['processing_parameters']`` -> empty dict.
    """
    processing = config.get("processing")
    if isinstance(processing, dict):
        if PROCESSING_PARAMETERS_KEY in processing:
            return processing[PROCESSING_PARAMETERS_KEY]
        # Some configs put the parameters directly under 'processing'
        return processing

    # Fall back to old structure: config['processing_parameters']
    if PROCESSING_PARAMETERS_KEY in config:
        return config[PROCESSING_PARAMETERS_KEY]

    logger.warning("Could not find processing_parameters in config. Using empty dict.")
    return {}


def get_nested_param(
    processing_config: dict,
    keys: List[str],
    expected_type=None,
    is_required: bool = True,
    default: Optional[float] = None,
):
    """Safely fetch and validate a nested value from the processing config.

    Args:
        processing_config: The resolved ``processing_parameters`` dict.
        keys: List of keys for nested access.
        expected_type: Expected type of the value.
        is_required: Whether the parameter is required.
        default: Default value if not found and not required.

    Returns:
        The value at the nested path.

    Raises:
        USPConfigError: If a required parameter is missing or has the wrong type.
    """
    value = processing_config
    path_str = "processing_config"
    for key in keys:
        path_str += f"['{key}']"
        if isinstance(value, dict):
            value = value.get(key)
            if value is None:
                if is_required:
                    msg = (
                        f"Missing required configuration parameter at path: {path_str}"
                    )
                    logger.error(msg)
                    raise USPConfigError(msg)
                else:
                    logger.debug(
                        f"Optional config parameter not found at path: {path_str}. Using default: {default}"
                    )
                    return default
        else:  # Path is broken
            if is_required:
                msg = (
                    f"Invalid config structure. Expected dict at path before {path_str}"
                )
                logger.error(msg)
                raise USPConfigError(msg)
            else:
                return default

    # Type check if required and type provided
    if (
        is_required
        and expected_type is not None
        and not isinstance(value, expected_type)
    ):
        # Allow int to satisfy float requirement
        if expected_type is float and isinstance(value, int):
            value = float(value)
        else:
            msg = (
                f"Config parameter at path config{path_str} has wrong type. "
                f"Expected {expected_type.__name__}, got {type(value).__name__}"
            )
            logger.error(msg)
            raise USPConfigError(msg)

    # Specific check for numeric > 0 if needed (e.g., density)
    if expected_type in [float, int] and isinstance(value, (int, float)) and value <= 0:
        # Check if this parameter *needs* to be positive
        if any(p in keys for p in _POSITIVE_PARAM_HINTS):
            msg = f"Numeric config parameter at path config{path_str} must be positive, got {value}"
            logger.error(msg)
            raise USPConfigError(msg)

    return value


def get_first(config: dict, paths, default=None):
    """Return the first non-None value found by walking any of ``paths`` in ``config``.

    ``paths`` is a list of key-lists; the first path that resolves to a
    non-None value wins. Used for the multi-layout config lookups in the
    carbon-balance and rates/yields calculations.
    """
    for path in paths:
        value = config
        ok = True
        for key in path:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                ok = False
                break
        if ok and value is not None:
            return value
    return default
