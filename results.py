"""
results.py
Pretty-print the comparison table of results.
"""

from statistics import Stats


def print_results(shape_name:    str,
                  analytical_v:  float,
                  point_stats:   Stats,
                  line_stats:    Stats):
    """
    Print a formatted comparison table.

    Example output:

    ══════════════════════════════════════════════════════════════════
    Shape : Sphere (r=1.0)
    ══════════════════════════════════════════════════════════════════
    Method              Volume        Rel. Error     FOM          Time (s)
    ──────────────────────────────────────────────────────────────────
    Analytical          4.188790      —              —            —
    Point sampling      4.191234      0.318 %        18234.1      0.432
    Line drawing        4.188901      0.105 %        56217.4      0.381
    ──────────────────────────────────────────────────────────────────
    Most efficient  :  Line drawing  (FOM ratio 3.08×)
    Error vs analyt.:  Point  0.057 %   |  Line  0.002 %
    ══════════════════════════════════════════════════════════════════
    """
    W = 70

    def fmt_vol(v):
        return f"{v:.6f}"

    def fmt_err(e):
        if e == float('inf'):
            return "inf"
        return f"{e * 100:.4f} %"

    def fmt_fom(f):
        if f == float('inf'):
            return "inf"
        return f"{f:.2f}"

    def fmt_time(t):
        return f"{t:.4f} s"

    def pct_diff(estimated, reference):
        if abs(reference) < 1e-14:
            return float('inf')
        return abs(estimated - reference) / reference * 100.0

    print()
    print("═" * W)
    print(f"  Shape : {shape_name}")
    print("═" * W)

    # Header row
    col = "{:<20} {:<14} {:<14} {:<14} {:<10}"
    print(col.format("Method", "Volume", "Rel. Error", "FOM", "Time"))
    print("─" * W)

    # Analytical row
    print(col.format(
        "Analytical",
        fmt_vol(analytical_v),
        "—",
        "—",
        "—"
    ))

    # Point sampling row
    print(col.format(
        f"Point ({point_stats.n:,})",
        fmt_vol(point_stats.volume),
        fmt_err(point_stats.rel_err),
        fmt_fom(point_stats.fom),
        fmt_time(point_stats.cpu_time),
    ))

    # Line drawing row
    print(col.format(
        f"Line ({line_stats.n:,})",
        fmt_vol(line_stats.volume),
        fmt_err(line_stats.rel_err),
        fmt_fom(line_stats.fom),
        fmt_time(line_stats.cpu_time),
    ))

    print("─" * W)

    # Which algorithm has higher FOM?
    if point_stats.fom >= line_stats.fom:
        winner      = "Point sampling"
        fom_ratio   = point_stats.fom / max(line_stats.fom, 1e-14)
    else:
        winner      = "Line drawing"
        fom_ratio   = line_stats.fom / max(point_stats.fom, 1e-14)

    print(f"  Most efficient   : {winner}  (FOM ratio {fom_ratio:.2f}×)")

    # Error vs analytical reference
    p_diff = pct_diff(point_stats.volume, analytical_v)
    l_diff = pct_diff(line_stats.volume,  analytical_v)
    print(f"  Error vs analyt. : Point {p_diff:.4f} %   |   Line {l_diff:.4f} %")

    print("═" * W)
    print()