def deep_update(original, updates):
    """
    Recursively update nested dictionaries without overwriting entire sections.
    """
    for key, value in updates.items():
        if isinstance(value, dict) and key in original and isinstance(original[key], dict):
            deep_update(original[key], value)
        else:
            original[key] = value
    return original