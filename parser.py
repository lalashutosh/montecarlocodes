"""
parser.py
Reads and validates the .dat input file.

Format:
    key    value    # optional comment
    shape  sphere
    r      1.0
    ...
"""

REQUIRED_KEYS = {
    'sphere':  ['r', 'n_points', 'n_lines'],
    'shell':   ['r', 'n_points', 'n_lines'],
    'cone':    ['r', 'h', 'n_points', 'n_lines'],
    'frustum': ['r', 'r_top', 'h', 'n_points', 'n_lines'],
}

VALID_SHAPES = set(REQUIRED_KEYS.keys())

# Which keys are ints vs floats
INT_KEYS   = {'n_points', 'n_lines'}
FLOAT_KEYS = {'r', 'h', 'r_top'}


def parse_input(filepath: str) -> dict:
    """
    Parse a .dat input file and return a validated config dict.

    Returns
    -------
    dict with keys: shape (str), r (float), h (float, optional),
                    r_top (float, optional), n_points (int), n_lines (int)

    Raises
    ------
    FileNotFoundError  : if the file does not exist
    ValueError         : if required keys are missing or values are invalid
    """
    raw = {}

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: '{filepath}'")

    for lineno, line in enumerate(lines, start=1):
        # Strip inline comments
        line = line.split('#')[0].strip()

        # Skip blank lines
        if not line:
            continue

        parts = line.split()

        if len(parts) < 2:
            print(f"  [Warning] Line {lineno}: cannot parse '{line}', skipping.")
            continue

        key   = parts[0].lower()
        value = parts[1]

        if key in raw:
            print(f"  [Warning] Line {lineno}: duplicate key '{key}', overwriting.")

        raw[key] = value

    # --- Validate shape ---
    if 'shape' not in raw:
        raise ValueError("Missing required key 'shape' in input file.")

    shape = raw['shape'].lower()
    if shape not in VALID_SHAPES:
        raise ValueError(
            f"Unknown shape '{shape}'. "
            f"Valid options: {', '.join(sorted(VALID_SHAPES))}"
        )

    # --- Build typed config dict ---
    config = {'shape': shape}

    # Parse all numeric keys present in file
    for key, value in raw.items():
        if key == 'shape':
            continue
        if key in INT_KEYS:
            try:
                config[key] = int(value)
            except ValueError:
                raise ValueError(f"Key '{key}' must be an integer, got '{value}'.")
        elif key in FLOAT_KEYS:
            try:
                config[key] = float(value)
            except ValueError:
                raise ValueError(f"Key '{key}' must be a float, got '{value}'.")
        else:
            print(f"  [Warning] Unknown key '{key}' in input file, ignoring.")

    # --- Check all required keys are present for this shape ---
    for key in REQUIRED_KEYS[shape]:
        if key not in config:
            raise ValueError(
                f"Missing required key '{key}' for shape '{shape}'."
            )

    # --- Sanity checks on values ---
    if config.get('r', 1) <= 0:
        raise ValueError(f"Radius 'r' must be positive, got {config['r']}.")
    if 'h' in config and config['h'] <= 0:
        raise ValueError(f"Height 'h' must be positive, got {config['h']}.")
    if 'r_top' in config and config['r_top'] < 0:
        raise ValueError(f"Top radius 'r_top' must be non-negative, got {config['r_top']}.")
    if 'r_top' in config and config['r_top'] >= config['r']:
        raise ValueError(
            f"Top radius 'r_top' ({config['r_top']}) must be less than "
            f"base radius 'r' ({config['r']})."
        )
    if config.get('n_points', 1) <= 0:
        raise ValueError(f"'n_points' must be a positive integer.")
    if config.get('n_lines', 1) <= 0:
        raise ValueError(f"'n_lines' must be a positive integer.")

    return config


def print_config(config: dict):
    """Pretty-print the parsed configuration."""
    print("=" * 40)
    print("  Input Configuration")
    print("=" * 40)
    for key, value in config.items():
        print(f"  {key:<12} {value}")
    print("=" * 40)