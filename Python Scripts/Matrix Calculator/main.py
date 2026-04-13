"""Matrix Calculator — CLI tool.

Add, subtract, multiply, transpose, determinant, inverse,
and row-reduce matrices.  Supports arbitrary integer/float entries
with pretty-printed output.

Usage:
    python main.py
"""

import math


# ---------------------------------------------------------------------------
# Matrix type alias and helpers
# ---------------------------------------------------------------------------

Matrix = list[list[float]]


def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0] * cols for _ in range(rows)]


def identity(n: int) -> Matrix:
    m = zeros(n, n)
    for i in range(n):
        m[i][i] = 1.0
    return m


def shape(m: Matrix) -> tuple[int, int]:
    return len(m), len(m[0]) if m else 0


def copy_matrix(m: Matrix) -> Matrix:
    return [row[:] for row in m]


def add(a: Matrix, b: Matrix) -> Matrix:
    r, c = shape(a)
    return [[a[i][j] + b[i][j] for j in range(c)] for i in range(r)]


def subtract(a: Matrix, b: Matrix) -> Matrix:
    r, c = shape(a)
    return [[a[i][j] - b[i][j] for j in range(c)] for i in range(r)]


def scalar_multiply(s: float, m: Matrix) -> Matrix:
    return [[s * v for v in row] for row in m]


def multiply(a: Matrix, b: Matrix) -> Matrix:
    ra, ca = shape(a)
    rb, cb = shape(b)
    result = zeros(ra, cb)
    for i in range(ra):
        for k in range(ca):
            for j in range(cb):
                result[i][j] += a[i][k] * b[k][j]
    return result


def transpose(m: Matrix) -> Matrix:
    r, c = shape(m)
    return [[m[i][j] for i in range(r)] for j in range(c)]


def determinant(m: Matrix) -> float:
    n, _ = shape(m)
    if n == 1:
        return m[0][0]
    if n == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]
    mat = copy_matrix(m)
    sign = 1.0
    for col in range(n):
        # Partial pivoting
        max_row = max(range(col, n), key=lambda r: abs(mat[r][col]))
        if max_row != col:
            mat[col], mat[max_row] = mat[max_row], mat[col]
            sign = -sign
        if abs(mat[col][col]) < 1e-12:
            return 0.0
        for row in range(col + 1, n):
            factor = mat[row][col] / mat[col][col]
            for k in range(col, n):
                mat[row][k] -= factor * mat[col][k]
    det = sign
    for i in range(n):
        det *= mat[i][i]
    return det


def inverse(m: Matrix) -> Matrix | None:
    n, _ = shape(m)
    aug = [row[:] + identity(n)[i] for i, row in enumerate(m)]
    for col in range(n):
        max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[max_row] = aug[max_row], aug[col]
        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            return None  # singular
        aug[col] = [v / pivot for v in aug[col]]
        for row in range(n):
            if row != col:
                factor = aug[row][col]
                aug[row] = [aug[row][k] - factor * aug[col][k] for k in range(2 * n)]
    return [row[n:] for row in aug]


def rref(m: Matrix) -> Matrix:
    """Reduced row echelon form."""
    mat = copy_matrix(m)
    r, c = shape(mat)
    pivot_row = 0
    for col in range(c):
        max_r = None
        for row in range(pivot_row, r):
            if abs(mat[row][col]) > 1e-12:
                if max_r is None or abs(mat[row][col]) > abs(mat[max_r][col]):
                    max_r = row
        if max_r is None:
            continue
        mat[pivot_row], mat[max_r] = mat[max_r], mat[pivot_row]
        pivot = mat[pivot_row][col]
        mat[pivot_row] = [v / pivot for v in mat[pivot_row]]
        for row in range(r):
            if row != pivot_row:
                factor = mat[row][col]
                mat[row] = [mat[row][k] - factor * mat[pivot_row][k] for k in range(c)]
        pivot_row += 1
    return mat


def trace(m: Matrix) -> float:
    n = min(shape(m))
    return sum(m[i][i] for i in range(n))


# ---------------------------------------------------------------------------
# Display & input helpers
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    if v == int(v) and abs(v) < 1e9:
        return str(int(v))
    return f"{v:.4g}"


def print_matrix(m: Matrix, label: str = "") -> None:
    if label:
        print(f"\n  {label}")
    if not m:
        print("  (empty)")
        return
    # Compute column widths
    str_vals = [[_fmt(v) for v in row] for row in m]
    widths   = [max(len(row[j]) for row in str_vals) for j in range(len(str_vals[0]))]
    for row in str_vals:
        cells = "  ".join(v.rjust(widths[j]) for j, v in enumerate(row))
        print(f"  │ {cells} │")


def read_matrix(name: str = "matrix") -> Matrix | None:
    print(f"  Enter {name} row by row (blank line when done):")
    rows = []
    while True:
        line = input("  > ").strip()
        if not line:
            break
        try:
            row = [float(v) for v in line.replace(",", " ").split()]
            rows.append(row)
        except ValueError:
            print("  Invalid numbers.")
            return None
    if not rows:
        return None
    # Validate uniform row length
    c = len(rows[0])
    if any(len(r) != c for r in rows):
        print("  All rows must have the same length.")
        return None
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Matrix Calculator
-----------------
1. Add / Subtract
2. Scalar multiply
3. Matrix multiply
4. Transpose
5. Determinant
6. Inverse
7. Reduced Row Echelon Form
8. Trace
0. Quit
"""


def main() -> None:
    print("Matrix Calculator")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            a = read_matrix("matrix A")
            b = read_matrix("matrix B")
            if not a or not b:
                continue
            if shape(a) != shape(b):
                print("  Matrices must have the same dimensions.")
                continue
            op = input("  (a)dd or (s)ubtract? ").strip().lower()
            if op.startswith("s"):
                print_matrix(subtract(a, b), "A - B =")
            else:
                print_matrix(add(a, b), "A + B =")

        elif choice == "2":
            m = read_matrix()
            if not m:
                continue
            s_str = input("  Scalar: ").strip()
            try:
                s = float(s_str)
            except ValueError:
                print("  Invalid scalar.")
                continue
            print_matrix(scalar_multiply(s, m), f"{s} × A =")

        elif choice == "3":
            a = read_matrix("matrix A")
            b = read_matrix("matrix B")
            if not a or not b:
                continue
            ra, ca = shape(a)
            rb, cb = shape(b)
            if ca != rb:
                print(f"  Incompatible: A is {ra}×{ca}, B is {rb}×{cb}")
                continue
            print_matrix(multiply(a, b), "A × B =")

        elif choice == "4":
            m = read_matrix()
            if not m:
                continue
            print_matrix(transpose(m), "Aᵀ =")

        elif choice == "5":
            m = read_matrix()
            if not m:
                continue
            r, c = shape(m)
            if r != c:
                print("  Determinant requires a square matrix.")
                continue
            d = determinant(m)
            print(f"\n  det(A) = {_fmt(d)}")

        elif choice == "6":
            m = read_matrix()
            if not m:
                continue
            r, c = shape(m)
            if r != c:
                print("  Inverse requires a square matrix.")
                continue
            inv = inverse(m)
            if inv is None:
                print("  Matrix is singular (no inverse).")
            else:
                print_matrix(inv, "A⁻¹ =")

        elif choice == "7":
            m = read_matrix()
            if not m:
                continue
            print_matrix(rref(m), "RREF =")

        elif choice == "8":
            m = read_matrix()
            if not m:
                continue
            print(f"\n  trace(A) = {_fmt(trace(m))}")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
