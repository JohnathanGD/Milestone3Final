"""NFL team logos and colors from nflverse via nfl_data_py."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import colorsys
import re

import pandas as pd

from .constants import TEAM_ABBR_NORMALIZE, TEAM_FULL

NFLVERSE_LOGOS_CSV = (
    "https://raw.githubusercontent.com/nflverse/nflverse-pbp/master/teams_colors_logos.csv"
)

# Curated vivid accents when nflverse primaries are dark or clash in UI bars
DISPLAY_COLOR_OVERRIDES: dict[str, str] = {
    "ARI": "#FFB612",
    "ATL": "#A71930",
    "BAL": "#9E7C0C",
    "BUF": "#C60C30",
    "CAR": "#0085CA",
    "CHI": "#E64100",
    "CIN": "#FB4F14",
    "CLE": "#FF3C00",
    "DAL": "#869397",
    "DEN": "#FB4F14",
    "DET": "#0076B6",
    "GB": "#FFB612",
    "HOU": "#A71930",
    "IND": "#A5ACAF",
    "JAX": "#006778",
    "KC": "#FFB612",
    "LA": "#FFD100",
    "LAC": "#FFC20E",
    "LV": "#A5ACAF",
    "MIA": "#F58220",
    "MIN": "#FFC62F",
    "NE": "#C60C30",
    "NO": "#D3BC8D",
    "NYG": "#A71930",
    "NYJ": "#FFFFFF",
    "PHI": "#A5ACAF",
    "PIT": "#FFB612",
    "SEA": "#69BE28",
    "SF": "#B3995D",
    "TB": "#FF7900",
    "TEN": "#4B92DB",
    "WAS": "#FFB612",
}


@dataclass(frozen=True)
class TeamBrand:
    abbr: str
    name: str
    color: str
    color2: str
    color3: str
    color4: str
    logo_espn: str
    logo_squared: str
    logo_wikipedia: str


def _parse_hex(hex_color: str) -> tuple[float, float, float] | None:
    if not hex_color or not isinstance(hex_color, str):
        return None
    value = hex_color.strip()
    if not value.startswith("#"):
        return None
    value = value.lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if not re.fullmatch(r"[0-9a-fA-F]{6}", value):
        return None
    r = int(value[0:2], 16) / 255.0
    g = int(value[2:4], 16) / 255.0
    b = int(value[4:6], 16) / 255.0
    return r, g, b


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(max(0, min(255, round(r * 255)))),
        int(max(0, min(255, round(g * 255)))),
        int(max(0, min(255, round(b * 255)))),
    )


def _luminance(rgb: tuple[float, float, float]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _color_score(rgb: tuple[float, float, float]) -> float:
    """Higher = better for UI bars (vivid, not too dark/light)."""
    r, g, b = rgb
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if l < 0.12 or l > 0.88:
        return s * 0.35
    if s < 0.18:
        return s * 0.5
    return s * 1.25 + (0.5 - abs(l - 0.48))


def _boost_color(hex_color: str, sat_mult: float = 1.45, min_sat: float = 0.52) -> str:
    rgb = _parse_hex(hex_color)
    if not rgb:
        return "#6b7280"
    r, g, b = rgb
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s = min(1.0, max(min_sat, s * sat_mult))
    l = min(0.62, max(0.32, l * 0.92 + 0.06))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return _rgb_to_hex(r2, g2, b2)


def _pick_palette_color(colors: list[str]) -> str:
    best_hex = "#6b7280"
    best_score = -1.0
    for raw in colors:
        rgb = _parse_hex(raw)
        if not rgb:
            continue
        score = _color_score(rgb)
        if score > best_score:
            best_score = score
            best_hex = raw
    return best_hex


@lru_cache(maxsize=1)
def load_teams_df() -> pd.DataFrame:
    try:
        import nfl_data_py as nfl

        df = nfl.import_team_desc()
    except Exception:
        df = pd.read_csv(NFLVERSE_LOGOS_CSV)

    df = df.copy()
    df["team_abbr"] = df["team_abbr"].astype(str)
    return df


@lru_cache(maxsize=1)
def load_team_brands() -> dict[str, TeamBrand]:
    df = load_teams_df()
    brands: dict[str, TeamBrand] = {}

    for _, row in df.iterrows():
        abbr = str(row["team_abbr"])
        norm = TEAM_ABBR_NORMALIZE.get(abbr, abbr)
        brand = TeamBrand(
            abbr=norm,
            name=str(row["team_name"]),
            color=str(row.get("team_color", "#333333") or "#333333"),
            color2=str(row.get("team_color2", "") or ""),
            color3=str(row.get("team_color3", "") or ""),
            color4=str(row.get("team_color4", "") or ""),
            logo_espn=str(row.get("team_logo_espn", "") or ""),
            logo_squared=str(row.get("team_logo_squared", "") or ""),
            logo_wikipedia=str(row.get("team_logo_wikipedia", "") or ""),
        )
        if norm not in brands:
            brands[norm] = brand

    return brands


def _name_to_abbr() -> dict[str, str]:
    out = {name: abbr for abbr, name in TEAM_FULL.items()}
    df = load_teams_df()
    for _, row in df.iterrows():
        out[str(row["team_name"])] = TEAM_ABBR_NORMALIZE.get(
            str(row["team_abbr"]), str(row["team_abbr"])
        )
    return out


@lru_cache(maxsize=1)
def name_to_abbr_map() -> dict[str, str]:
    return _name_to_abbr()


def resolve_abbr(team_label: str) -> str | None:
    if not team_label or (isinstance(team_label, float) and pd.isna(team_label)):
        return None
    label = str(team_label).strip()
    if label.upper() in TEAM_FULL or label.upper() in TEAM_ABBR_NORMALIZE:
        raw = label.upper()
        return TEAM_ABBR_NORMALIZE.get(raw, raw)
    return name_to_abbr_map().get(label)


def _raw_palette_for_abbr(abbr: str) -> list[str]:
    brand = load_team_brands().get(abbr)
    if not brand:
        return []
    return [c for c in [brand.color, brand.color2, brand.color3, brand.color4] if c]


def _color_distance(a: str, b: str) -> float:
    ra, ga, ba = _parse_hex(a) or (0, 0, 0)
    rb, gb, bb = _parse_hex(b) or (0, 0, 0)
    return ((ra - rb) ** 2 + (ga - gb) ** 2 + (ba - bb) ** 2) ** 0.5


def team_display_color(team_label: str) -> str:
    """Bold, UI-friendly team color (boosted accent from nflverse palette)."""
    abbr = resolve_abbr(team_label)
    if not abbr:
        return "#6b7280"

    if abbr in DISPLAY_COLOR_OVERRIDES:
        base = DISPLAY_COLOR_OVERRIDES[abbr]
    else:
        base = _pick_palette_color(_raw_palette_for_abbr(abbr))

    boosted = _boost_color(base)
    if abbr == "NYJ" and _luminance(_parse_hex(boosted) or (1, 1, 1)) > 0.75:
        return "#125740"
    return boosted


def matchup_display_colors(home: str, away: str) -> tuple[str, str]:
    """Return distinct boosted colors for home/away so bars do not blend."""
    home_abbr = resolve_abbr(home)
    away_abbr = resolve_abbr(away)
    home_c = team_display_color(home)
    away_c = team_display_color(away)

    if _color_distance(home_c, away_c) >= 0.22:
        return home_c, away_c

    # Too similar — use alternate accent from nflverse for the away side
    if away_abbr:
        alts = _raw_palette_for_abbr(away_abbr)
        primary = load_team_brands().get(away_abbr)
        if primary:
            used = {home_c.lower(), team_display_color(home).lower()}
            for alt in alts:
                if alt.lower() in used:
                    continue
                candidate = _boost_color(alt, sat_mult=1.55, min_sat=0.58)
                if _color_distance(home_c, candidate) >= 0.18:
                    away_c = candidate
                    break

    if _color_distance(home_c, away_c) < 0.18 and home_abbr and away_abbr:
        home_c = _boost_color(home_c, sat_mult=1.65, min_sat=0.6)
        away_c = _boost_color(away_c, sat_mult=1.65, min_sat=0.6)

    return home_c, away_c


def logo_url(team_label: str, prefer: str = "espn") -> str | None:
    abbr = resolve_abbr(team_label)
    if not abbr:
        return None
    brand = load_team_brands().get(abbr)
    if not brand:
        return None
    if prefer == "squared" and brand.logo_squared:
        return brand.logo_squared
    if prefer == "wikipedia" and brand.logo_wikipedia:
        return brand.logo_wikipedia
    return brand.logo_espn or brand.logo_squared or brand.logo_wikipedia or None


def team_color(team_label: str) -> str:
    """Alias for labels/text — uses the same vivid display palette."""
    return team_display_color(team_label)


def brand_for(team_label: str) -> TeamBrand | None:
    abbr = resolve_abbr(team_label)
    if not abbr:
        return None
    return load_team_brands().get(abbr)
