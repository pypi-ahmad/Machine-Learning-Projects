"""Physics Formulas Calculator — CLI tool.

Compute common physics quantities: kinematics, Newton's laws,
energy/work/power, waves, ideal gas law, and electrical circuits.

Usage:
    python main.py
"""

import math


# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------

G   = 9.80665   # m/s²  (standard gravity)
C   = 3e8       # m/s   (speed of light)
R   = 8.314     # J/(mol·K) (gas constant)
k_B = 1.381e-23 # J/K (Boltzmann constant)
h   = 6.626e-34 # J·s (Planck constant)


# ---------------------------------------------------------------------------
# Kinematics (SUVAT)
# ---------------------------------------------------------------------------

def kinematics(s=None, u=None, v=None, a=None, t=None):
    """Solve for any one SUVAT unknown given 3 knowns."""
    known = sum(x is not None for x in (s, u, v, a, t))
    if known < 3:
        return None, "Need at least 3 knowns."
    results = {}
    try:
        if s is None and u is not None and v is not None and t is not None:
            s = (u + v) / 2 * t
            results["s"] = s
        if v is None and u is not None and a is not None and t is not None:
            v = u + a * t
            results["v"] = v
        if s is None and u is not None and a is not None and t is not None:
            s = u * t + 0.5 * a * t ** 2
            results["s"] = s
        if v is None and u is not None and a is not None and s is not None:
            disc = u ** 2 + 2 * a * s
            v = math.sqrt(disc) if disc >= 0 else complex(0, math.sqrt(-disc))
            results["v"] = v
        if t is None and a is not None and u is not None and v is not None and a != 0:
            t = (v - u) / a
            results["t"] = t
        if a is None and s is not None and u is not None and t is not None and t != 0:
            a = 2 * (s - u * t) / t ** 2
            results["a"] = a
    except Exception as e:
        return None, str(e)
    return results or {"s": s, "u": u, "v": v, "a": a, "t": t}, None


# ---------------------------------------------------------------------------
# Newton's Laws
# ---------------------------------------------------------------------------

def force(m: float, a: float) -> float:           return m * a       # F = ma
def weight(m: float) -> float:                    return m * G       # W = mg
def friction(mu: float, N: float) -> float:       return mu * N      # f = μN
def centripetal(m: float, v: float, r: float):    return m*v**2/r    # F = mv²/r


# ---------------------------------------------------------------------------
# Energy, Work, Power
# ---------------------------------------------------------------------------

def kinetic_energy(m: float, v: float) -> float:  return 0.5 * m * v ** 2
def potential_energy(m: float, h: float) -> float: return m * G * h
def work(F: float, d: float, angle_deg: float = 0) -> float:
    return F * d * math.cos(math.radians(angle_deg))
def power(W: float, t: float) -> float:           return W / t if t else 0
def efficiency(useful: float, total: float) -> float:
    return useful / total * 100 if total else 0


# ---------------------------------------------------------------------------
# Waves
# ---------------------------------------------------------------------------

def wave_speed(f: float, lam: float) -> float:    return f * lam     # v = fλ
def wave_freq(v: float, lam: float) -> float:     return v / lam
def wave_lambda(v: float, f: float) -> float:     return v / f
def doppler(f0: float, vs: float, vo: float, source_approach: bool = True) -> float:
    """Doppler effect: vs = source speed, vo = observer speed."""
    sign_s = 1 if source_approach else -1
    return f0 * (C + vo) / (C - sign_s * vs)


# ---------------------------------------------------------------------------
# Ideal Gas Law
# ---------------------------------------------------------------------------

def ideal_gas(P=None, V=None, n=None, T=None):
    """PV = nRT. Solve for missing variable."""
    if P is None and None not in (V, n, T):
        return {"P": n * R * T / V}
    if V is None and None not in (P, n, T):
        return {"V": n * R * T / P}
    if n is None and None not in (P, V, T):
        return {"n": P * V / (R * T)}
    if T is None and None not in (P, V, n):
        return {"T": P * V / (n * R)}
    return {}


# ---------------------------------------------------------------------------
# Electrical
# ---------------------------------------------------------------------------

def ohms_law(V=None, I=None, R_=None):
    """V = IR."""
    if V is None:
        return {"V": I * R_}
    if I is None:
        return {"I": V / R_}
    if R_ is None:
        return {"R": V / I}
    return {}


def electrical_power(V=None, I=None, R_=None):
    if None not in (V, I):
        return {"P": V * I}
    if None not in (I, R_):
        return {"P": I ** 2 * R_}
    if None not in (V, R_):
        return {"P": V ** 2 / R_}
    return {}


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if isinstance(v, complex):
        return f"{v}"
    return f"{v:.6g}"


def gf(prompt: str) -> float | None:
    s = input(prompt + " (blank=unknown): ").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Physics Formulas Calculator
----------------------------
1. Kinematics (SUVAT)
2. Forces (F=ma, weight, friction, centripetal)
3. Energy, Work, Power
4. Waves
5. Ideal Gas Law  (PV = nRT)
6. Electrical (Ohm's Law, Power)
7. Constants reference
0. Quit
"""


def main() -> None:
    print("Physics Formulas Calculator")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            print("  SUVAT: s=displacement, u=initial velocity, v=final velocity, a=acceleration, t=time")
            s = gf("  s (m)  ")
            u = gf("  u (m/s)")
            v = gf("  v (m/s)")
            a = gf("  a (m/s²)")
            t = gf("  t (s)  ")
            results, err = kinematics(s, u, v, a, t)
            if err:
                print(f"  {err}")
            else:
                print("\n  Results:")
                for k, val in results.items():
                    print(f"    {k} = {_fmt(val)}")

        elif choice == "2":
            print("  (f) F=ma  (w) Weight=mg  (r) Friction=μN  (c) Centripetal")
            sub = input("  Subtype: ").strip().lower()
            if sub == "f":
                m = gf("  mass m (kg)"); a_ = gf("  acceleration a (m/s²)")
                if None not in (m, a_):
                    print(f"\n  F = {_fmt(force(m, a_))} N")
            elif sub == "w":
                m = gf("  mass m (kg)")
                if m:
                    print(f"\n  Weight = {_fmt(weight(m))} N")
            elif sub == "r":
                mu = gf("  μ (friction coefficient)"); N = gf("  Normal force N (N)")
                if None not in (mu, N):
                    print(f"\n  Friction = {_fmt(friction(mu, N))} N")
            elif sub == "c":
                m = gf("  mass m (kg)"); v_ = gf("  speed v (m/s)"); r_ = gf("  radius r (m)")
                if None not in (m, v_, r_):
                    print(f"\n  Centripetal F = {_fmt(centripetal(m, v_, r_))} N")

        elif choice == "3":
            print("  (k) KE  (p) PE  (w) Work  (P) Power  (e) Efficiency")
            sub = input("  Subtype: ").strip()
            if sub == "k":
                m = gf("  mass (kg)"); v_ = gf("  speed (m/s)")
                if None not in (m, v_):
                    print(f"\n  KE = {_fmt(kinetic_energy(m, v_))} J")
            elif sub == "p":
                m = gf("  mass (kg)"); h_ = gf("  height (m)")
                if None not in (m, h_):
                    print(f"\n  PE = {_fmt(potential_energy(m, h_))} J")
            elif sub == "w":
                F_ = gf("  Force (N)"); d = gf("  distance (m)")
                ang = gf("  angle with displacement (°) [0]") or 0.0
                if None not in (F_, d):
                    print(f"\n  Work = {_fmt(work(F_, d, ang))} J")
            elif sub == "P":
                W_ = gf("  Work (J)"); t_ = gf("  time (s)")
                if None not in (W_, t_):
                    print(f"\n  Power = {_fmt(power(W_, t_))} W")
            elif sub == "e":
                u_ = gf("  Useful energy/power"); tot = gf("  Total energy/power")
                if None not in (u_, tot):
                    print(f"\n  Efficiency = {_fmt(efficiency(u_, tot))} %")

        elif choice == "4":
            print("  v = fλ  (solve for v/f/λ)")
            sub = input("  Solve for (v/f/L): ").strip().lower()
            if sub == "v":
                f_ = gf("  frequency f (Hz)"); lam = gf("  wavelength λ (m)")
                if None not in (f_, lam):
                    print(f"\n  v = {_fmt(wave_speed(f_, lam))} m/s")
            elif sub == "f":
                v_ = gf("  speed v (m/s)"); lam = gf("  wavelength λ (m)")
                if None not in (v_, lam):
                    print(f"\n  f = {_fmt(wave_freq(v_, lam))} Hz")
            elif sub == "l":
                v_ = gf("  speed v (m/s)"); f_ = gf("  frequency f (Hz)")
                if None not in (v_, f_):
                    print(f"\n  λ = {_fmt(wave_lambda(v_, f_))} m")

        elif choice == "5":
            print("  Leave unknown blank.")
            P  = gf("  P (Pa) "); V_ = gf("  V (m³)"); n_ = gf("  n (mol)")
            T_ = gf("  T (K) ")
            result = ideal_gas(P, V_, n_, T_)
            if result:
                for k, val in result.items():
                    print(f"\n  {k} = {_fmt(val)}")
            else:
                print("  Not enough knowns.")

        elif choice == "6":
            print("  (o) Ohm's Law (V=IR)  (p) Power")
            sub = input("  Subtype: ").strip().lower()
            if sub == "o":
                V_ = gf("  V (V)"); I_ = gf("  I (A)"); R__ = gf("  R (Ω)")
                result = ohms_law(V_, I_, R__)
                for k, val in result.items():
                    print(f"\n  {k} = {_fmt(val)}")
            elif sub == "p":
                V_ = gf("  V (V)"); I_ = gf("  I (A)"); R__ = gf("  R (Ω)")
                result = electrical_power(V_, I_, R__)
                for k, val in result.items():
                    print(f"\n  {k} = {_fmt(val)} W")

        elif choice == "7":
            print(f"\n  g  = {G} m/s²  (standard gravity)")
            print(f"  c  = {C:.2e} m/s  (speed of light)")
            print(f"  R  = {R} J/(mol·K)  (gas constant)")
            print(f"  kB = {k_B:.3e} J/K  (Boltzmann)")
            print(f"  h  = {h:.3e} J·s  (Planck)")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
