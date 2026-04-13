"""Geometry Calculator — CLI tool.

Calculate area, perimeter, volume, and surface area for common
2D and 3D shapes.  Also includes coordinate geometry helpers.

Usage:
    python main.py
"""

import math


# ---------------------------------------------------------------------------
# 2D shapes
# ---------------------------------------------------------------------------

def circle(r: float) -> dict:
    return {"area": math.pi * r ** 2,
            "circumference": 2 * math.pi * r,
            "diameter": 2 * r}


def rectangle(w: float, h: float) -> dict:
    return {"area": w * h,
            "perimeter": 2 * (w + h),
            "diagonal": math.hypot(w, h)}


def square(s: float) -> dict:
    return rectangle(s, s) | {"side": s}


def triangle_sides(a: float, b: float, c: float) -> dict:
    """By 3 sides (Heron's formula)."""
    s = (a + b + c) / 2
    disc = s * (s - a) * (s - b) * (s - c)
    if disc < 0:
        return {"error": "Invalid triangle (sides don't satisfy triangle inequality)"}
    area = math.sqrt(disc)
    return {"area": area,
            "perimeter": a + b + c,
            "semi-perimeter": s,
            "is_equilateral": a == b == c,
            "is_isosceles": (a == b or b == c or a == c),
            "is_right": any(
                abs(x ** 2 - y ** 2 - z ** 2) < 1e-9
                for x, y, z in [(a, b, c), (b, a, c), (c, a, b)]
            )}


def triangle_base_height(b: float, h: float) -> dict:
    return {"area": 0.5 * b * h}


def trapezoid(a: float, b: float, h: float) -> dict:
    return {"area": 0.5 * (a + b) * h,
            "parallel sides": (a, b),
            "height": h}


def parallelogram(base: float, height: float, side: float = 0.0) -> dict:
    res = {"area": base * height}
    if side:
        res["perimeter"] = 2 * (base + side)
    return res


def regular_polygon(n: int, side: float) -> dict:
    area = (n * side ** 2) / (4 * math.tan(math.pi / n))
    return {"sides": n,
            "area": area,
            "perimeter": n * side,
            "interior_angle_deg": (n - 2) * 180 / n}


def ellipse(a: float, b: float) -> dict:
    # Ramanujan approximation for perimeter
    h = (a - b) ** 2 / (a + b) ** 2
    perim = math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))
    return {"area": math.pi * a * b,
            "perimeter_approx": perim,
            "semi-major": a,
            "semi-minor": b}


# ---------------------------------------------------------------------------
# 3D shapes
# ---------------------------------------------------------------------------

def sphere(r: float) -> dict:
    return {"volume": 4 / 3 * math.pi * r ** 3,
            "surface_area": 4 * math.pi * r ** 2}


def cylinder(r: float, h: float) -> dict:
    return {"volume": math.pi * r ** 2 * h,
            "lateral_area": 2 * math.pi * r * h,
            "total_surface": 2 * math.pi * r * (r + h)}


def cone(r: float, h: float) -> dict:
    slant = math.hypot(r, h)
    return {"volume": math.pi * r ** 2 * h / 3,
            "slant_height": slant,
            "lateral_area": math.pi * r * slant,
            "total_surface": math.pi * r * (r + slant)}


def cube(s: float) -> dict:
    return {"volume": s ** 3,
            "surface_area": 6 * s ** 2,
            "space_diagonal": s * math.sqrt(3)}


def cuboid(l: float, w: float, h: float) -> dict:
    return {"volume": l * w * h,
            "surface_area": 2 * (l * w + w * h + l * h),
            "space_diagonal": math.sqrt(l ** 2 + w ** 2 + h ** 2)}


def torus(R: float, r: float) -> dict:
    return {"volume": 2 * math.pi ** 2 * R * r ** 2,
            "surface_area": 4 * math.pi ** 2 * R * r}


# ---------------------------------------------------------------------------
# Coordinate geometry
# ---------------------------------------------------------------------------

def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def midpoint(x1, y1, x2, y2):
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1) if x2 != x1 else float("inf")


def circle_from_center_point(cx, cy, px, py):
    r = distance(cx, cy, px, py)
    return circle(r)


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------

def print_results(d: dict) -> None:
    print()
    for k, v in d.items():
        if isinstance(v, float):
            print(f"  {k:<22}: {v:.6g}")
        else:
            print(f"  {k:<22}: {v}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU_2D = """
  2D Shapes:
  c) Circle      r) Rectangle   s) Square    t) Triangle (sides)
  T) Triangle (base×height)     z) Trapezoid p) Parallelogram
  n) Regular polygon             e) Ellipse
"""

MENU_3D = """
  3D Shapes:
  S) Sphere      C) Cylinder    K) Cone      B) Cube      X) Cuboid
  O) Torus
"""

MENU_COORD = """
  Coordinate Geometry:
  d) Distance    m) Midpoint    l) Slope
"""

MENU = """
Geometry Calculator
-------------------
1. 2D shapes
2. 3D shapes
3. Coordinate geometry
0. Quit
"""


def gf(prompt: str) -> float | None:
    try:
        return float(input(prompt).strip())
    except ValueError:
        print("  Invalid number.")
        return None


def main() -> None:
    print("Geometry Calculator")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            print(MENU_2D)
            sub = input("  Shape: ").strip()
            if sub == "c":
                r = gf("  Radius: ")
                if r is not None:
                    print_results(circle(r))
            elif sub == "r":
                w = gf("  Width: "); h = gf("  Height: ")
                if None not in (w, h):
                    print_results(rectangle(w, h))
            elif sub == "s":
                s = gf("  Side: ")
                if s is not None:
                    print_results(square(s))
            elif sub == "t":
                a = gf("  Side a: "); b = gf("  Side b: "); c = gf("  Side c: ")
                if None not in (a, b, c):
                    print_results(triangle_sides(a, b, c))
            elif sub == "T":
                b = gf("  Base: "); h = gf("  Height: ")
                if None not in (b, h):
                    print_results(triangle_base_height(b, h))
            elif sub == "z":
                a = gf("  Parallel side a: "); b = gf("  Parallel side b: ")
                h = gf("  Height: ")
                if None not in (a, b, h):
                    print_results(trapezoid(a, b, h))
            elif sub == "p":
                base = gf("  Base: "); height = gf("  Height: ")
                side = gf("  Side (0 if unknown): ") or 0
                if None not in (base, height):
                    print_results(parallelogram(base, height, side))
            elif sub == "n":
                n_s = input("  Number of sides: ").strip()
                side = gf("  Side length: ")
                try:
                    n = int(n_s)
                    if side is not None:
                        print_results(regular_polygon(n, side))
                except ValueError:
                    print("  Invalid n.")
            elif sub == "e":
                a = gf("  Semi-major axis a: "); b = gf("  Semi-minor axis b: ")
                if None not in (a, b):
                    print_results(ellipse(a, b))
            else:
                print("  Unknown shape.")

        elif choice == "2":
            print(MENU_3D)
            sub = input("  Shape: ").strip()
            if sub == "S":
                r = gf("  Radius: ")
                if r is not None:
                    print_results(sphere(r))
            elif sub == "C":
                r = gf("  Radius: "); h = gf("  Height: ")
                if None not in (r, h):
                    print_results(cylinder(r, h))
            elif sub == "K":
                r = gf("  Radius: "); h = gf("  Height: ")
                if None not in (r, h):
                    print_results(cone(r, h))
            elif sub == "B":
                s = gf("  Side: ")
                if s is not None:
                    print_results(cube(s))
            elif sub == "X":
                l = gf("  Length: "); w = gf("  Width: "); h = gf("  Height: ")
                if None not in (l, w, h):
                    print_results(cuboid(l, w, h))
            elif sub == "O":
                R = gf("  Major radius R: "); r = gf("  Minor radius r: ")
                if None not in (R, r):
                    print_results(torus(R, r))
            else:
                print("  Unknown shape.")

        elif choice == "3":
            print(MENU_COORD)
            sub = input("  Operation: ").strip().lower()
            x1 = gf("  x1: "); y1 = gf("  y1: ")
            x2 = gf("  x2: "); y2 = gf("  y2: ")
            if None in (x1, y1, x2, y2):
                continue
            if sub == "d":
                print(f"\n  Distance = {distance(x1,y1,x2,y2):.6g}")
            elif sub == "m":
                mx, my = midpoint(x1,y1,x2,y2)
                print(f"\n  Midpoint = ({mx:.4g}, {my:.4g})")
            elif sub == "l":
                s = slope(x1,y1,x2,y2)
                print(f"\n  Slope = {s:.6g}")
                if s != float("inf"):
                    b = y1 - s * x1
                    print(f"  Line: y = {s:.4g}x + {b:.4g}")
            else:
                print("  Unknown operation.")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
