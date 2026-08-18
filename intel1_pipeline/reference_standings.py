from __future__ import annotations

from typing import Any


REFERENCE_DRIVER_STANDINGS_2026: list[dict[str, Any]] = [
    {"driver": "Kimi Antonelli", "driver_id": "andrea_kimi_antonelli", "team": "Mercedes", "team_id": "mercedes", "position": 1, "points": 219.0, "wins": 6},
    {"driver": "Lewis Hamilton", "driver_id": "lewis_hamilton", "team": "Ferrari", "team_id": "ferrari", "position": 2, "points": 169.0, "wins": 1},
    {"driver": "George Russell", "driver_id": "george_russell", "team": "Mercedes", "team_id": "mercedes", "position": 3, "points": 160.0, "wins": 2},
    {"driver": "Charles Leclerc", "driver_id": "charles_leclerc", "team": "Ferrari", "team_id": "ferrari", "position": 4, "points": 138.0, "wins": 1},
    {"driver": "Lando Norris", "driver_id": "lando_norris", "team": "McLaren", "team_id": "mclaren", "position": 5, "points": 128.0, "wins": 1},
    {"driver": "Max Verstappen", "driver_id": "max_verstappen", "team": "Red Bull Racing", "team_id": "red_bull", "position": 6, "points": 109.0, "wins": 0},
    {"driver": "Oscar Piastri", "driver_id": "oscar_piastri", "team": "McLaren", "team_id": "mclaren", "position": 7, "points": 92.0, "wins": 0},
    {"driver": "Isack Hadjar", "driver_id": "isack_hadjar", "team": "Red Bull Racing", "team_id": "red_bull", "position": 8, "points": 68.0, "wins": 0},
    {"driver": "Liam Lawson", "driver_id": "liam_lawson", "team": "Racing Bulls", "team_id": "racing_bulls", "position": 9, "points": 43.0, "wins": 0},
    {"driver": "Pierre Gasly", "driver_id": "pierre_gasly", "team": "Alpine", "team_id": "alpine", "position": 10, "points": 42.0, "wins": 0},
    {"driver": "Arvid Lindblad", "driver_id": "arvid_lindblad", "team": "Racing Bulls", "team_id": "racing_bulls", "position": 11, "points": 23.0, "wins": 0},
    {"driver": "Franco Colapinto", "driver_id": "franco_colapinto", "team": "Alpine", "team_id": "alpine", "position": 12, "points": 19.0, "wins": 0},
    {"driver": "Oliver Bearman", "driver_id": "oliver_bearman", "team": "Haas F1 Team", "team_id": "haas", "position": 13, "points": 18.0, "wins": 0},
    {"driver": "Gabriel Bortoleto", "driver_id": "gabriel_bortoleto", "team": "Audi", "team_id": "audi", "position": 14, "points": 10.0, "wins": 0},
    {"driver": "Carlos Sainz", "driver_id": "carlos_sainz", "team": "Williams", "team_id": "williams", "position": 15, "points": 6.0, "wins": 0},
    {"driver": "Alexander Albon", "driver_id": "alexander_albon", "team": "Williams", "team_id": "williams", "position": 16, "points": 5.0, "wins": 0},
    {"driver": "Esteban Ocon", "driver_id": "esteban_ocon", "team": "Haas F1 Team", "team_id": "haas", "position": 17, "points": 3.0, "wins": 0},
    {"driver": "Nico Hulkenberg", "driver_id": "nico_hulkenberg", "team": "Audi", "team_id": "audi", "position": 18, "points": 2.0, "wins": 0},
    {"driver": "Fernando Alonso", "driver_id": "fernando_alonso", "team": "Aston Martin", "team_id": "aston_martin", "position": 19, "points": 1.0, "wins": 0},
    {"driver": "Lance Stroll", "driver_id": "lance_stroll", "team": "Aston Martin", "team_id": "aston_martin", "position": 20, "points": 0.0, "wins": 0},
    {"driver": "Valtteri Bottas", "driver_id": "valtteri_bottas", "team": "Cadillac", "team_id": "cadillac", "position": 21, "points": 0.0, "wins": 0},
    {"driver": "Sergio Perez", "driver_id": "sergio_perez", "team": "Cadillac", "team_id": "cadillac", "position": 22, "points": 0.0, "wins": 0},
]

REFERENCE_CONSTRUCTOR_STANDINGS_2026: list[dict[str, Any]] = [
    {"team": "Mercedes", "team_id": "mercedes", "position": 1, "points": 379.0, "wins": 8},
    {"team": "Ferrari", "team_id": "ferrari", "position": 2, "points": 307.0, "wins": 2},
    {"team": "McLaren", "team_id": "mclaren", "position": 3, "points": 220.0, "wins": 1},
    {"team": "Red Bull Racing", "team_id": "red_bull", "position": 4, "points": 177.0, "wins": 0},
    {"team": "Racing Bulls", "team_id": "racing_bulls", "position": 5, "points": 66.0, "wins": 0},
    {"team": "Alpine", "team_id": "alpine", "position": 6, "points": 61.0, "wins": 0},
    {"team": "Haas F1 Team", "team_id": "haas", "position": 7, "points": 21.0, "wins": 0},
    {"team": "Audi", "team_id": "audi", "position": 8, "points": 12.0, "wins": 0},
    {"team": "Williams", "team_id": "williams", "position": 9, "points": 11.0, "wins": 0},
    {"team": "Aston Martin", "team_id": "aston_martin", "position": 10, "points": 1.0, "wins": 0},
    {"team": "Cadillac", "team_id": "cadillac", "position": 11, "points": 0.0, "wins": 0},
]

def reference_driver_standings(year: int | None = None) -> list[dict[str, Any]]:
    return [item.copy() for item in REFERENCE_DRIVER_STANDINGS_2026] if year in {None, 2026} else []


def reference_constructor_standings(year: int | None = None) -> list[dict[str, Any]]:
    return [item.copy() for item in REFERENCE_CONSTRUCTOR_STANDINGS_2026] if year in {None, 2026} else []
