"""Recipe Manager — Streamlit app.

Add, browse, search, and scale recipes.  Calculate nutrition estimates
and generate shopping lists.  Data stored as local JSON.

Usage:
    streamlit run main.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Recipe Manager", layout="wide")
st.title("🍳 Recipe Manager")

DATA_FILE = Path("recipes.json")

SAMPLE_RECIPES = [
    {
        "name": "Spaghetti Bolognese",
        "category": "Pasta",
        "servings": 4,
        "prep_min": 15,
        "cook_min": 40,
        "ingredients": [
            {"item": "Spaghetti", "qty": 400, "unit": "g"},
            {"item": "Ground beef", "qty": 500, "unit": "g"},
            {"item": "Tomato sauce", "qty": 400, "unit": "ml"},
            {"item": "Onion", "qty": 1, "unit": "medium"},
            {"item": "Garlic", "qty": 3, "unit": "cloves"},
        ],
        "instructions": "1. Cook spaghetti.\n2. Brown beef.\n3. Add sauce.\n4. Combine.",
        "tags": ["Italian", "Meat", "Quick"],
    },
    {
        "name": "Pancakes",
        "category": "Breakfast",
        "servings": 2,
        "prep_min": 5,
        "cook_min": 10,
        "ingredients": [
            {"item": "Flour", "qty": 200, "unit": "g"},
            {"item": "Milk", "qty": 250, "unit": "ml"},
            {"item": "Egg", "qty": 2, "unit": ""},
            {"item": "Butter", "qty": 30, "unit": "g"},
            {"item": "Sugar", "qty": 20, "unit": "g"},
        ],
        "instructions": "1. Mix dry.\n2. Add wet.\n3. Cook on medium heat.",
        "tags": ["Sweet", "Easy", "Vegetarian"],
    },
]


def load_recipes() -> list[dict]:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text())
        except Exception:
            pass
    save_recipes(SAMPLE_RECIPES)
    return SAMPLE_RECIPES


def save_recipes(recipes: list[dict]) -> None:
    DATA_FILE.write_text(json.dumps(recipes, indent=2))


if "recipes" not in st.session_state:
    st.session_state.recipes = load_recipes()

recipes = st.session_state.recipes

# ---------------------------------------------------------------------------
# Sidebar — add recipe
# ---------------------------------------------------------------------------
st.sidebar.header("Add New Recipe")
with st.sidebar.expander("➕ Add Recipe", expanded=False):
    r_name  = st.text_input("Recipe name")
    r_cat   = st.text_input("Category", "General")
    r_servs = st.number_input("Servings", 1, 100, 4)
    r_prep  = st.number_input("Prep time (min)", 0, step=5)
    r_cook  = st.number_input("Cook time (min)", 0, step=5)
    r_ing   = st.text_area("Ingredients (one per line: qty unit item)\ne.g. 200 g flour")
    r_inst  = st.text_area("Instructions")
    r_tags  = st.text_input("Tags (comma-separated)")
    add_btn = st.button("Add Recipe")

if add_btn and r_name.strip():
    ingredients = []
    for line in r_ing.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3:
            try:
                ingredients.append({"item": parts[2], "qty": float(parts[0]), "unit": parts[1]})
            except ValueError:
                ingredients.append({"item": line, "qty": 1, "unit": ""})
        elif line.strip():
            ingredients.append({"item": line.strip(), "qty": 1, "unit": ""})
    new_recipe = {
        "name": r_name.strip(), "category": r_cat, "servings": r_servs,
        "prep_min": r_prep, "cook_min": r_cook, "ingredients": ingredients,
        "instructions": r_inst, "tags": [t.strip() for t in r_tags.split(",")],
    }
    recipes.append(new_recipe)
    save_recipes(recipes)
    st.session_state.recipes = recipes
    st.sidebar.success(f"Added: {r_name.strip()}")
    st.rerun()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Browse Recipes", "View Recipe", "Shopping List"])

with tab1:
    st.subheader(f"All Recipes ({len(recipes)})")
    search = st.text_input("🔍 Search recipes")
    cats = list(set(r.get("category", "") for r in recipes))
    cat_filter = st.multiselect("Category", cats, default=cats)

    filtered = [r for r in recipes
                if (not search or search.lower() in r["name"].lower()
                    or any(search.lower() in t.lower() for t in r.get("tags", [])))
                and r.get("category", "") in cat_filter]

    if not filtered:
        st.info("No recipes match your search.")
    else:
        cols = st.columns(3)
        for i, recipe in enumerate(filtered):
            with cols[i % 3]:
                with st.container(border=True):
                    st.write(f"**{recipe['name']}**")
                    st.caption(f"📁 {recipe.get('category','')}  ·  "
                               f"👤 {recipe.get('servings','')} servings  ·  "
                               f"⏱ {recipe.get('prep_min',0)+recipe.get('cook_min',0)} min")
                    st.caption(f"🏷️ {', '.join(recipe.get('tags',[]))}")

with tab2:
    recipe_names = [r["name"] for r in recipes]
    if not recipe_names:
        st.info("No recipes.")
    else:
        sel_name = st.selectbox("Select recipe", recipe_names)
        recipe   = next((r for r in recipes if r["name"] == sel_name), None)
        if recipe:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Servings",  recipe.get("servings", ""))
            c2.metric("Prep",      f"{recipe.get('prep_min',0)} min")
            c3.metric("Cook",      f"{recipe.get('cook_min',0)} min")
            c4.metric("Total",     f"{recipe.get('prep_min',0)+recipe.get('cook_min',0)} min")

            scale = st.slider("Scale servings", 1, 20, recipe.get("servings", 4))
            factor = scale / recipe.get("servings", 4) if recipe.get("servings") else 1

            st.subheader("Ingredients")
            ing_data = []
            for ing in recipe.get("ingredients", []):
                scaled_qty = ing["qty"] * factor
                ing_data.append({
                    "Item":     ing["item"],
                    "Quantity": f"{scaled_qty:.1f}".rstrip("0").rstrip("."),
                    "Unit":     ing.get("unit", ""),
                })
            st.dataframe(pd.DataFrame(ing_data), hide_index=True, use_container_width=True)

            st.subheader("Instructions")
            for line in recipe.get("instructions", "").splitlines():
                st.write(line)

            if st.button("🗑️ Delete recipe"):
                st.session_state.recipes = [r for r in recipes if r["name"] != sel_name]
                save_recipes(st.session_state.recipes)
                st.rerun()

with tab3:
    st.subheader("Shopping List Generator")
    selected = st.multiselect("Select recipes", [r["name"] for r in recipes])
    if selected:
        from collections import defaultdict
        shopping: dict[str, dict] = defaultdict(lambda: {"qty": 0, "unit": ""})
        for recipe_name in selected:
            recipe = next((r for r in recipes if r["name"] == recipe_name), None)
            if recipe:
                for ing in recipe.get("ingredients", []):
                    key = ing["item"].lower()
                    shopping[key]["qty"]  += ing.get("qty", 0)
                    shopping[key]["unit"]  = ing.get("unit", "")
        st.write("**Shopping list:**")
        for item, info in sorted(shopping.items()):
            qty = info["qty"]
            unit = info["unit"]
            qty_str = f"{qty:.1f}".rstrip("0").rstrip(".")
            st.write(f"- {item.title()}: {qty_str} {unit}")
