# src/data/config/loader.py
import yaml
from pathlib import Path
import logging
from copy import deepcopy
from typing import Union, Dict

from ..core.validation import validate_config_structure
from ..core.exceptions import USPConfigError

logger = logging.getLogger(__name__)

# Path to the default configuration file within the package
DEFAULT_CONFIG_PATH = Path(__file__).parent / 'default_config.yaml'

def merge_configs(default: dict, custom: dict) -> dict:
    """
    Recursively merges the custom config dict into the default dict.
    Custom values override default values.
    Lists are overwritten, not merged.
    """
    merged = deepcopy(default)
    for key, value in custom.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            # If it's not a dictionary or key doesn't exist in default,
            # simply overwrite/add the custom value.
            merged[key] = value
    return merged

def load_config(experiment_config_path: Union[str, Path, None] = None) -> dict:
    """
    Loads the default configuration and merges an optional experiment-specific
    configuration file on top of it.

    Args:
        experiment_config_path: Path to the experiment-specific YAML config file.
                                If None, only the default config is returned.

    Returns:
        A dictionary representing the final merged configuration.
    """
    # Load default config
    if not DEFAULT_CONFIG_PATH.exists():
        logger.error(f"Default configuration file not found at {DEFAULT_CONFIG_PATH}")
        raise FileNotFoundError(f"Default configuration file not found at {DEFAULT_CONFIG_PATH}")

    try:
        with open(DEFAULT_CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded default configuration from {DEFAULT_CONFIG_PATH}")
    except Exception as e:
        logger.error(f"Error loading default configuration: {e}")
        raise

    # Load and merge experiment config if provided
    if experiment_config_path:
        exp_path = Path(experiment_config_path)
        if exp_path.exists():
            try:
                with open(exp_path, 'r', encoding='utf-8') as f:
                    experiment_config = yaml.safe_load(f)
                    if experiment_config: # Ensure file is not empty
                        logger.info(f"Loading and merging experiment config from {exp_path}")
                        config = merge_configs(config, experiment_config)
                    else:
                        logger.warning(f"Experiment config file is empty: {exp_path}. Using defaults.")
            except Exception as e:
                logger.error(f"Error loading experiment configuration from {exp_path}: {e}. Using defaults.")
        else:
            logger.warning(f"Experiment config file not found: {exp_path}. Using defaults.")
    else:
         logger.info("No experiment-specific configuration path provided. Using defaults.")


    # Load optional media config (e.g., delisa_media_config.yaml)
    media_config_name = None
    if isinstance(config, dict):
        media_config_name = config.get('metadata', {}).get('media_config_file')
        if not media_config_name:
            media_config_name = config.get('analysis', {}).get('media_config_file')

    media_overrides = None
    if isinstance(config, dict):
        media_overrides = config.get('analysis', {}).get('delisa_media_config')

    if media_config_name:
        candidates = []
        if isinstance(media_config_name, str):
            media_path = Path(media_config_name)
            if media_path.is_absolute():
                candidates.append(media_path)
            if experiment_config_path:
                candidates.append(Path(experiment_config_path).parent / media_config_name)
            # Fallback to bundled media config inside the package
            candidates.append(Path(__file__).resolve().parent.parent / 'analysis' / media_config_name)

        media_config_path = next((p for p in candidates if p.exists()), None)
        if media_config_path:
            try:
                with open(media_config_path, 'r', encoding='utf-8') as f:
                    media_config = yaml.safe_load(f)
                if isinstance(media_config, dict):
                    config.setdefault('analysis', {})
                    if isinstance(media_overrides, dict) and media_overrides:
                        media_config = merge_configs(media_config, media_overrides)
                    config['analysis']['delisa_media_config'] = media_config
                    logger.info(f"Loaded media config from {media_config_path}")
                else:
                    logger.warning(f"Media config at {media_config_path} is empty or invalid.")
            except Exception as e:
                logger.error(f"Error loading media config from {media_config_path}: {e}")
        else:
            logger.warning(f"Media config file not found: {media_config_name}")

    # Validate config structure after all merging is complete
    if isinstance(config, dict):
        try:
            validate_config_structure(config)
        except USPConfigError as e:
            logger.error(f"Config validation failed: {e}")
            raise

    return config

# Example usage within this module (for testing)
if __name__ == '__main__':
    print("--- Testing Default Config Loading ---")
    default_cfg = load_config()
    print(default_cfg)

    print("\n--- Testing Non-existent Experiment Config ---")
    cfg_non_existent = load_config("non_existent_config.yaml")
    # Should be the same as default_cfg

    print("\n--- Testing With Example Experiment Config ---")
    # Create a dummy experiment config for testing
    dummy_exp_path = Path("./dummy_exp_config.yaml")
    dummy_exp_data = {
        'parsing': {'bioreactor': {'decimal_separator': '.'}},
        'processing': {'feed_params': {'start_volume_ml': 1200}}
    }
    with open(dummy_exp_path, 'w') as f:
        yaml.dump(dummy_exp_data, f)

    merged_cfg = load_config(dummy_exp_path)
    print("Merged Config:")
    print(merged_cfg)
    print(f"Decimal Separator (merged): {merged_cfg['parsing']['bioreactor']['decimal_separator']}") # Should be '.'
    print(f"Start Volume (merged): {merged_cfg['processing']['feed_params']['start_volume_ml']}")     # Should be 1200
    print(f"Column Separator (default): {merged_cfg['parsing']['bioreactor']['column_separator']}") # Should be ';'

    dummy_exp_path.unlink() # Clean up dummy file
