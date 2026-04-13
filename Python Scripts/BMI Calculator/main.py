"""BMI Calculator — CLI tool.

Calculates Body Mass Index (BMI) using metric or imperial units.
Shows BMI category, healthy weight range, and basic guidance.

Usage:
    python main.py
"""

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BMI_CATEGORIES = [
    (0,    18.5, "Underweight",       "You may need to gain weight. Consult a doctor."),
    (18.5, 25.0, "Normal weight",     "You are in the healthy weight range."),
    (25.0, 30.0, "Overweight",        "Consider a balanced diet and more exercise."),
    (30.0, 35.0, "Obese (Class I)",   "Health risks are increased. Consult a doctor."),
    (35.0, 40.0, "Obese (Class II)",  "Significant health risks. Seek medical advice."),
    (40.0, 999,  "Obese (Class III)", "Severe health risks. Immediate medical advice."),
]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def bmi_metric(weight_kg: float, height_m: float) -> float:
    if height_m <= 0:
        raise ValueError("Height must be positive.")
    if weight_kg <= 0:
        raise ValueError("Weight must be positive.")
    return weight_kg / (height_m ** 2)


def bmi_imperial(weight_lb: float, height_in: float) -> float:
    """BMI formula for US customary units."""
    if height_in <= 0:
        raise ValueError("Height must be positive.")
    if weight_lb <= 0:
        raise ValueError("Weight must be positive.")
    return (weight_lb / (height_in ** 2)) * 703


def classify_bmi(bmi: float) -> tuple[str, str]:
    for low, high, label, advice in BMI_CATEGORIES:
        if low <= bmi < high:
            return label, advice
    return "Unknown", ""


def healthy_weight_range(height_m: float) -> tuple[float, float]:
    """Return (min_kg, max_kg) for normal BMI range at given height."""
    min_w = 18.5 * height_m ** 2
    max_w = 24.9 * height_m ** 2
    return min_w, max_w


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def get_positive_float(prompt: str) -> float:
    while True:
        try:
            val = float(input(prompt).strip())
            if val > 0:
                return val
            print("  Value must be greater than zero.")
        except ValueError:
            print("  Please enter a valid number.")


def main() -> None:
    print("BMI Calculator")
    print("=" * 40)

    while True:
        print("\n1. Metric  (kg / cm)")
        print("2. Imperial  (lb / ft + in)")
        print("0. Quit")
        choice = input("\nChoice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            weight = get_positive_float("  Weight (kg): ")
            height_cm = get_positive_float("  Height (cm): ")
            height_m = height_cm / 100
            try:
                bmi = bmi_metric(weight, height_m)
                label, advice = classify_bmi(bmi)
                min_w, max_w = healthy_weight_range(height_m)
                print(f"\n  BMI             : {bmi:.2f}")
                print(f"  Category        : {label}")
                print(f"  Advice          : {advice}")
                print(f"  Healthy range   : {min_w:.1f} – {max_w:.1f} kg at {height_cm:.0f} cm")
            except ValueError as e:
                print(f"  Error: {e}")

        elif choice == "2":
            weight_lb = get_positive_float("  Weight (lb): ")
            feet = get_positive_float("  Height (feet): ")
            inches_extra = get_positive_float("  Height (extra inches, 0 if none): ")
            total_inches = feet * 12 + inches_extra
            try:
                bmi = bmi_imperial(weight_lb, total_inches)
                label, advice = classify_bmi(bmi)
                height_m = total_inches * 0.0254
                min_w_kg, max_w_kg = healthy_weight_range(height_m)
                min_w_lb = min_w_kg * 2.20462
                max_w_lb = max_w_kg * 2.20462
                print(f"\n  BMI             : {bmi:.2f}")
                print(f"  Category        : {label}")
                print(f"  Advice          : {advice}")
                print(f"  Healthy range   : {min_w_lb:.1f} – {max_w_lb:.1f} lb")
            except ValueError as e:
                print(f"  Error: {e}")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
