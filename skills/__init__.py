"""Local skill install directory.

Skill bundles are installed here from the remote catalog via ``unclaw onboard``
or ``unclaw skills install``.  This directory is intentionally empty in a fresh
checkout — no skills are bundled with the main repo.

Each installed subdirectory (e.g. ``skills/weather/``) is excluded from git
by the project ``.gitignore`` rule ``skills/*/``.
"""

__all__: list[str] = []
