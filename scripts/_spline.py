"""Catmull-Rom -> cubic Bezier conversion, so we can define a profile as simple
landmark points and get a smooth path automatically, instead of hand-guessing
bezier control points (which is what produced blob-shaped garbage before)."""


def catmull_rom_to_bezier(points, closed=True):
    """points: list of (x,y). Returns an SVG path 'd' string through them."""
    pts = points[:]
    n = len(pts)
    if closed:
        p_ext = [pts[-1]] + pts + [pts[0], pts[1]]
    else:
        p_ext = [pts[0]] + pts + [pts[-1]]

    d = f"M {pts[0][0]:.2f},{pts[0][1]:.2f} "
    count = n if closed else n - 1
    for i in range(count):
        p0 = p_ext[i]
        p1 = p_ext[i + 1]
        p2 = p_ext[i + 2]
        p3 = p_ext[i + 3]
        c1x = p1[0] + (p2[0] - p0[0]) / 6
        c1y = p1[1] + (p2[1] - p0[1]) / 6
        c2x = p2[0] - (p3[0] - p1[0]) / 6
        c2y = p2[1] - (p3[1] - p1[1]) / 6
        d += f"C {c1x:.2f},{c1y:.2f} {c2x:.2f},{c2y:.2f} {p2[0]:.2f},{p2[1]:.2f} "
    if closed:
        d += "Z"
    return d
