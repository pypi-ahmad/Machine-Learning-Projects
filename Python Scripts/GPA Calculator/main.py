"""GPA Calculator — CLI tool.

Supports two grading scales:
  - 4.0 US GPA scale (A=4, B=3, C=2, D=1, F=0)
  - Weighted GPA (accounts for course credit hours)

Usage:
    python main.py
"""

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GRADE_POINTS: dict[str, float] = {
    "A+": 4.0, "A": 4.0, "A-": 3.7,
    "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7,
    "D+": 1.3, "D": 1.0, "D-": 0.7,
    "F": 0.0,
}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def letter_to_points(letter: str) -> float:
    key = letter.strip().upper()
    if key not in GRADE_POINTS:
        raise ValueError(f"Unknown grade: '{letter}'. Valid grades: {', '.join(GRADE_POINTS)}")
    return GRADE_POINTS[key]


def calculate_gpa(courses: list[tuple[str, float, float]]) -> dict:
    """
    courses: list of (grade_letter, credit_hours, optional_weight_multiplier)
    Returns dict with gpa, total_credits, total_points, letter_grade.
    """
    if not courses:
        raise ValueError("No courses entered.")

    total_points = 0.0
    total_credits = 0.0
    for grade, credits, weight in courses:
        pts = letter_to_points(grade)
        total_points += pts * credits * weight
        total_credits += credits * weight

    if total_credits == 0:
        raise ValueError("Total credit hours cannot be zero.")

    gpa = total_points / total_credits

    # GPA to letter
    if gpa >= 3.7:
        letter = "A" if gpa >= 4.0 else "A-"
    elif gpa >= 3.3:
        letter = "B+"
    elif gpa >= 3.0:
        letter = "B"
    elif gpa >= 2.7:
        letter = "B-"
    elif gpa >= 2.3:
        letter = "C+"
    elif gpa >= 2.0:
        letter = "C"
    elif gpa >= 1.0:
        letter = "D"
    else:
        letter = "F"

    return {
        "gpa": round(gpa, 3),
        "letter": letter,
        "total_credits": total_credits,
        "total_points": round(total_points, 3),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_float(prompt: str, min_val: float = 0.0) -> float:
    while True:
        try:
            val = float(input(prompt).strip())
            if val < min_val:
                print(f"  Value must be >= {min_val}")
                continue
            return val
        except ValueError:
            print("  Please enter a valid number.")


def enter_courses() -> list[tuple[str, float, float]]:
    courses = []
    print("\nEnter courses. Press Enter with no grade to finish.\n")
    i = 1
    while True:
        grade_str = input(f"  Course {i} grade (e.g. A, B+, C-): ").strip()
        if not grade_str:
            break
        try:
            letter_to_points(grade_str)  # validate
        except ValueError as e:
            print(f"  Error: {e}")
            continue

        credits = get_float("  Credit hours: ", min_val=0.5)
        weight_str = input("  Weight multiplier (1.0 = normal, 1.5 = honors, leave blank=1.0): ").strip()
        weight = 1.0
        if weight_str:
            try:
                weight = float(weight_str)
                if weight <= 0:
                    print("  Weight must be > 0, using 1.0.")
                    weight = 1.0
            except ValueError:
                print("  Invalid weight, using 1.0.")

        courses.append((grade_str.upper(), credits, weight))
        i += 1

    return courses


def print_grade_scale() -> None:
    print("\nGrade scale:")
    for g, pts in GRADE_POINTS.items():
        print(f"  {g:<4} = {pts:.1f}")


def main() -> None:
    print("GPA Calculator")
    print("=" * 40)

    while True:
        print("\n1. Calculate GPA")
        print("2. Show grade scale")
        print("0. Quit")
        choice = input("\nChoice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            courses = enter_courses()
            if not courses:
                print("  No courses entered.")
                continue
            try:
                result = calculate_gpa(courses)
                print(f"\n  {'─' * 30}")
                print(f"  Courses entered : {len(courses)}")
                print(f"  Total credits   : {result['total_credits']:.1f}")
                print(f"  Total points    : {result['total_points']:.3f}")
                print(f"  GPA             : {result['gpa']:.3f}  ({result['letter']})")
                print(f"  {'─' * 30}")
            except ValueError as e:
                print(f"  Error: {e}")

        elif choice == "2":
            print_grade_scale()

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
