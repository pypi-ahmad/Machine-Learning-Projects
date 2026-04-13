"""Trigonometry Helper — CLI tool.

Calculate trig functions, inverse trig, triangle properties,
unit circle values, and angle conversions.

Usage:
    python main.py
    python main.py --angle 45 --unit deg
    python main.py --triangle --sides 3 4 5
"""

import argparse
import math
import sys


DEG = math.pi / 180
RAD = 1.0


def fmt(n: float, dec: int = 6) -> str:
    return str(round(n, dec)).rstrip("0").rstrip(".")


def deg_to_rad(d: float) -> float: return d * DEG
def rad_to_deg(r: float) -> float: return r / DEG


def trig_values(angle: float, unit: str = "deg") -> None:
    """Compute all trig functions for a given angle."""
    if unit == "deg":
        rad = deg_to_rad(angle)
        deg = angle
    else:
        rad = angle
        deg = rad_to_deg(angle)

    print(f"\n  Angle: {fmt(deg)}°  =  {fmt(rad)} rad  =  {fmt(deg/400*360)}ᵍ (grad)")
    print(f"  {'─'*36}")

    funcs = [
        ("sin",  math.sin(rad)),
        ("cos",  math.cos(rad)),
        ("tan",  math.tan(rad) if abs(math.cos(rad)) > 1e-10 else None),
        ("csc",  1/math.sin(rad) if abs(math.sin(rad)) > 1e-10 else None),
        ("sec",  1/math.cos(rad) if abs(math.cos(rad)) > 1e-10 else None),
        ("cot",  math.cos(rad)/math.sin(rad) if abs(math.sin(rad)) > 1e-10 else None),
    ]
    for name, val in funcs:
        disp = fmt(val) if val is not None else "undefined"
        print(f"  {name:>4}({fmt(deg)}°) = {disp}")


def inverse_trig(value: float) -> None:
    """Compute all inverse trig functions for a value."""
    print(f"\n  Inverse trig for value = {value}")
    print(f"  {'─'*36}")
    if -1 <= value <= 1:
        rad = math.asin(value)
        print(f"  arcsin({value}) = {fmt(rad)} rad = {fmt(rad_to_deg(rad))}°")
        rad = math.acos(value)
        print(f"  arccos({value}) = {fmt(rad)} rad = {fmt(rad_to_deg(rad))}°")
    else:
        print("  arcsin/arccos: value out of range [-1, 1]")
    rad = math.atan(value)
    print(f"  arctan({value}) = {fmt(rad)} rad = {fmt(rad_to_deg(rad))}°")


def unit_circle(step: float = 30) -> None:
    """Display standard unit circle values."""
    angles = [0, 30, 45, 60, 90, 120, 135, 150, 180,
              210, 225, 240, 270, 300, 315, 330, 360]
    print(f"\n  {'Angle°':>8}  {'Radians':>12}  {'sin':>10}  {'cos':>10}  {'tan':>10}")
    print("  " + "─" * 58)
    for deg in angles:
        rad = deg_to_rad(deg)
        s   = math.sin(rad)
        c   = math.cos(rad)
        t   = s/c if abs(c) > 1e-10 else None
        ts  = fmt(t) if t is not None else "∞"
        # Pretty fractions of pi
        pi_map = {0:"0", 30:"π/6", 45:"π/4", 60:"π/3", 90:"π/2",
                  120:"2π/3", 135:"3π/4", 150:"5π/6", 180:"π",
                  210:"7π/6", 225:"5π/4", 240:"4π/3", 270:"3π/2",
                  300:"5π/3", 315:"7π/4", 330:"11π/6", 360:"2π"}
        rad_str = pi_map.get(deg, fmt(rad, 4))
        print(f"  {deg:>8}  {rad_str:>12}  {fmt(s,4):>10}  {fmt(c,4):>10}  {ts:>10}")


def triangle_solver(sides=None, angles=None) -> None:
    """Solve a triangle given SSS, SAS, or AAS/ASA."""
    print(f"\n  Triangle Solver")
    print(f"  {'─'*36}")
    if sides and len(sides) == 3:
        a, b, c = sides
        # Validate triangle inequality
        if a + b <= c or a + c <= b or b + c <= a:
            print("  Invalid triangle (triangle inequality fails).")
            return
        A = math.degrees(math.acos((b**2 + c**2 - a**2) / (2*b*c)))
        B = math.degrees(math.acos((a**2 + c**2 - b**2) / (2*a*c)))
        C = 180 - A - B
        area = 0.5 * a * b * math.sin(math.radians(C))
        perim = a + b + c
        print(f"  Sides:   a={fmt(a)}, b={fmt(b)}, c={fmt(c)}")
        print(f"  Angles:  A={fmt(A)}°, B={fmt(B)}°, C={fmt(C)}°")
        print(f"  Area:    {fmt(area)}")
        print(f"  Perimeter: {fmt(perim)}")
        s = perim / 2
        r = area / s                          # inradius
        R = a * b * c / (4 * area)            # circumradius
        print(f"  Inradius:     {fmt(r)}")
        print(f"  Circumradius: {fmt(R)}")
    else:
        print("  Provide 3 sides with --sides a b c")


def interactive():
    print("=== Trigonometry Helper ===")
    print("Commands: trig | inv | circle | triangle | convert | quit\n")
    while True:
        cmd = input("> ").strip().lower()
        if cmd in ("quit", "q", "exit"):
            break
        elif cmd == "trig":
            a = float(input("  Angle: ").strip())
            u = input("  Unit (deg/rad) [deg]: ").strip() or "deg"
            trig_values(a, u)
        elif cmd == "inv":
            v = float(input("  Value: ").strip())
            inverse_trig(v)
        elif cmd == "circle":
            unit_circle()
        elif cmd == "triangle":
            print("  Enter 3 sides (a b c):")
            sides = list(map(float, input("  > ").split()))
            triangle_solver(sides=sides)
        elif cmd == "convert":
            val  = float(input("  Value: ").strip())
            frm  = input("  From (deg/rad/grad): ").strip().lower()
            to   = input("  To   (deg/rad/grad): ").strip().lower()
            def to_rad(v, u):
                if u == "deg":  return v * math.pi / 180
                if u == "grad": return v * math.pi / 200
                return v
            def from_rad(v, u):
                if u == "deg":  return v * 180 / math.pi
                if u == "grad": return v * 200 / math.pi
                return v
            print(f"  {val} {frm} = {fmt(from_rad(to_rad(val, frm), to))} {to}")
        else:
            print("  Commands: trig | inv | circle | triangle | convert | quit")
        print()


def main():
    parser = argparse.ArgumentParser(description="Trigonometry Helper")
    parser.add_argument("--angle",    type=float, help="Angle value")
    parser.add_argument("--unit",     default="deg", choices=["deg","rad"],
                        help="Angle unit (default: deg)")
    parser.add_argument("--inv",      type=float, metavar="V",
                        help="Compute inverse trig for value V")
    parser.add_argument("--circle",   action="store_true", help="Show unit circle table")
    parser.add_argument("--triangle", action="store_true", help="Solve triangle")
    parser.add_argument("--sides",    nargs=3, type=float, metavar=("A","B","C"),
                        help="Three side lengths for triangle solver")
    args = parser.parse_args()

    if args.angle is not None:
        trig_values(args.angle, args.unit)
    elif args.inv is not None:
        inverse_trig(args.inv)
    elif args.circle:
        unit_circle()
    elif args.triangle or args.sides:
        triangle_solver(sides=args.sides)
    else:
        interactive()


if __name__ == "__main__":
    main()
