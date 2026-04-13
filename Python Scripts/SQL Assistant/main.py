"""SQL Assistant — Streamlit demo.

Interactive SQL query editor with an in-memory SQLite database,
schema explorer, query history, and natural-language-to-SQL hints.

Usage:
    streamlit run main.py
"""

import re
import sqlite3
import io

import pandas as pd
import streamlit as st

st.set_page_config(page_title="SQL Assistant", layout="wide")
st.title("🗄️ SQL Assistant")
st.caption("Write and execute SQL against a built-in SQLite database. Explore schema, view history, and get query hints.")


# ── In-memory SQLite database ─────────────────────────────────────────────────

@st.cache_resource
def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    _seed_database(conn)
    return conn


def _seed_database(conn: sqlite3.Connection):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS customers (
        id          INTEGER PRIMARY KEY,
        name        TEXT    NOT NULL,
        email       TEXT    UNIQUE,
        country     TEXT,
        signup_date TEXT,
        tier        TEXT    DEFAULT 'free'
    );

    CREATE TABLE IF NOT EXISTS products (
        id          INTEGER PRIMARY KEY,
        name        TEXT    NOT NULL,
        category    TEXT,
        price       REAL,
        stock       INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS orders (
        id          INTEGER PRIMARY KEY,
        customer_id INTEGER REFERENCES customers(id),
        order_date  TEXT,
        status      TEXT    DEFAULT 'pending',
        total       REAL
    );

    CREATE TABLE IF NOT EXISTS order_items (
        id          INTEGER PRIMARY KEY,
        order_id    INTEGER REFERENCES orders(id),
        product_id  INTEGER REFERENCES products(id),
        quantity    INTEGER,
        unit_price  REAL
    );

    CREATE TABLE IF NOT EXISTS employees (
        id          INTEGER PRIMARY KEY,
        name        TEXT    NOT NULL,
        department  TEXT,
        salary      REAL,
        hire_date   TEXT,
        manager_id  INTEGER REFERENCES employees(id)
    );

    INSERT OR IGNORE INTO customers VALUES
        (1,  'Alice Johnson',   'alice@email.com',  'US',  '2022-01-15', 'premium'),
        (2,  'Bob Smith',       'bob@email.com',    'UK',  '2022-03-22', 'free'),
        (3,  'Carol White',     'carol@email.com',  'CA',  '2022-06-10', 'premium'),
        (4,  'David Brown',     'david@email.com',  'US',  '2023-01-08', 'free'),
        (5,  'Eva Martinez',    'eva@email.com',    'DE',  '2023-04-17', 'premium'),
        (6,  'Frank Wilson',    'frank@email.com',  'AU',  '2023-07-25', 'free'),
        (7,  'Grace Lee',       'grace@email.com',  'US',  '2023-09-12', 'premium'),
        (8,  'Henry Davis',     'henry@email.com',  'FR',  '2024-01-03', 'free'),
        (9,  'Iris Clark',      'iris@email.com',   'US',  '2024-02-14', 'premium'),
        (10, 'Jack Turner',     'jack@email.com',   'JP',  '2024-03-21', 'free');

    INSERT OR IGNORE INTO products VALUES
        (1,  'Laptop Pro 15',      'Electronics',  1299.99, 45),
        (2,  'Wireless Mouse',     'Electronics',    29.99, 200),
        (3,  'USB-C Hub',          'Electronics',    49.99, 120),
        (4,  'Standing Desk',      'Furniture',     349.99, 15),
        (5,  'Ergonomic Chair',    'Furniture',     279.99, 22),
        (6,  'Python Book',        'Books',          39.99, 80),
        (7,  'Data Science Guide', 'Books',          54.99, 60),
        (8,  'Yoga Mat',           'Sports',         29.99, 150),
        (9,  'Protein Powder',     'Health',         44.99, 90),
        (10, 'Noise Earbuds',      'Electronics',   129.99, 55);

    INSERT OR IGNORE INTO orders VALUES
        (1,  1, '2024-01-10', 'delivered', 1329.98),
        (2,  2, '2024-01-15', 'delivered',   79.98),
        (3,  3, '2024-02-01', 'delivered',  349.99),
        (4,  4, '2024-02-20', 'shipped',    159.98),
        (5,  5, '2024-03-05', 'delivered',  629.98),
        (6,  1, '2024-03-18', 'pending',    129.99),
        (7,  7, '2024-04-01', 'delivered',  279.99),
        (8,  8, '2024-04-12', 'shipped',    214.97),
        (9,  9, '2024-04-20', 'pending',     89.98),
        (10, 3, '2024-05-01', 'delivered',   94.98);

    INSERT OR IGNORE INTO order_items VALUES
        (1,  1, 1, 1, 1299.99), (2,  1, 2, 1,   29.99),
        (3,  2, 2, 1,   29.99), (4,  2, 8, 1,   29.99), (5,  2, 9, 1,  44.99),
        (6,  3, 4, 1,  349.99),
        (7,  4, 6, 2,   39.99), (8,  4, 7, 1,   54.99), (9,  4, 8, 1,  29.99),
        (10, 5, 5, 1,  279.99), (11, 5, 3, 1,   49.99), (12, 5, 7, 1,  54.99), (13, 5, 9, 2,  44.99),
        (14, 6, 10,1,  129.99),
        (15, 7, 5, 1,  279.99),
        (16, 8, 6, 3,   39.99), (17, 8, 9, 2,   44.99), (18, 8, 2, 1,   29.99),
        (19, 9, 8, 2,   29.99), (20, 9, 2, 1,   29.99),
        (21,10, 6, 1,   39.99), (22,10, 9, 1,   44.99);

    INSERT OR IGNORE INTO employees VALUES
        (1, 'Sarah Chen',    'Engineering', 95000, '2020-03-01', NULL),
        (2, 'Mike Johnson',  'Engineering', 85000, '2021-06-15', 1),
        (3, 'Lisa Park',     'Marketing',   72000, '2021-01-10', NULL),
        (4, 'Tom Brown',     'Marketing',   65000, '2022-04-01', 3),
        (5, 'Anna White',    'Sales',       68000, '2020-11-20', NULL),
        (6, 'James Lee',     'Sales',       60000, '2022-08-05', 5),
        (7, 'Emily Davis',   'HR',          62000, '2021-09-14', NULL),
        (8, 'Carlos Ruiz',   'Engineering', 90000, '2020-07-22', 1),
        (9, 'Nina Patel',    'Sales',       63000, '2023-02-17', 5),
       (10, 'Oscar Kim',     'HR',          58000, '2023-05-08', 7);
    """)
    conn.commit()


# ── NL → SQL hint engine ──────────────────────────────────────────────────────

NL_HINTS = [
    (r"top\s+(\d+)\s+customers",
     "SELECT c.name, SUM(o.total) AS total_spent\nFROM customers c JOIN orders o ON c.id=o.customer_id\nGROUP BY c.id ORDER BY total_spent DESC LIMIT {n};"),
    (r"(total|revenue|sales)",
     "SELECT SUM(total) AS total_revenue FROM orders WHERE status='delivered';"),
    (r"(product|item).*(most|popular|top)",
     "SELECT p.name, SUM(oi.quantity) AS units_sold\nFROM order_items oi JOIN products p ON oi.product_id=p.id\nGROUP BY p.id ORDER BY units_sold DESC LIMIT 5;"),
    (r"(employee|staff).*(salary|earn)",
     "SELECT name, department, salary FROM employees ORDER BY salary DESC;"),
    (r"(order|purchase).*(status|pending|shipped)",
     "SELECT status, COUNT(*) AS count, SUM(total) AS value\nFROM orders GROUP BY status;"),
    (r"(low|out of).*(stock|inventory)",
     "SELECT name, category, stock FROM products WHERE stock < 30 ORDER BY stock;"),
    (r"customer.*(country|region|location)",
     "SELECT country, COUNT(*) AS customers FROM customers GROUP BY country ORDER BY customers DESC;"),
    (r"average.*(order|purchase)",
     "SELECT AVG(total) AS avg_order_value FROM orders;"),
    (r"(premium|tier)",
     "SELECT tier, COUNT(*) AS count FROM customers GROUP BY tier;"),
    (r"(department|team).*(average|avg|salary)",
     "SELECT department, AVG(salary) AS avg_salary FROM employees GROUP BY department ORDER BY avg_salary DESC;"),
]

def nl_to_sql_hint(text: str) -> str | None:
    text = text.lower()
    for pattern, template in NL_HINTS:
        m = re.search(pattern, text)
        if m:
            n = m.group(1) if m.lastindex and m.group(1).isdigit() else "5"
            return template.replace("{n}", n)
    return None


SAMPLE_QUERIES = {
    "All customers": "SELECT * FROM customers;",
    "All products": "SELECT * FROM products ORDER BY category, price;",
    "Orders with customer names": "SELECT o.id, c.name, o.order_date, o.status, o.total\nFROM orders o\nJOIN customers c ON o.customer_id = c.id\nORDER BY o.order_date DESC;",
    "Top products by revenue": "SELECT p.name, SUM(oi.quantity * oi.unit_price) AS revenue\nFROM order_items oi\nJOIN products p ON oi.product_id = p.id\nGROUP BY p.id\nORDER BY revenue DESC;",
    "Employee hierarchy": "SELECT e.name, e.department, e.salary, m.name AS manager\nFROM employees e\nLEFT JOIN employees m ON e.manager_id = m.id\nORDER BY e.department, e.salary DESC;",
    "Monthly order totals": "SELECT strftime('%Y-%m', order_date) AS month,\n       COUNT(*) AS orders,\n       SUM(total) AS revenue\nFROM orders\nGROUP BY month\nORDER BY month;",
    "Average order per customer tier": "SELECT c.tier, AVG(o.total) AS avg_order\nFROM orders o JOIN customers c ON o.customer_id = c.id\nGROUP BY c.tier;",
    "Low stock alert": "SELECT name, category, stock FROM products WHERE stock < 50 ORDER BY stock;",
}

# ── Session state ─────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []

conn = get_connection()

tab1, tab2, tab3, tab4 = st.tabs(["Query Editor", "Schema Explorer", "Query History", "NL Hints"])

with tab1:
    st.subheader("SQL Query Editor")

    preset = st.selectbox("Load sample query", ["(type your own)"] + list(SAMPLE_QUERIES.keys()))
    default_sql = SAMPLE_QUERIES.get(preset, "SELECT * FROM customers LIMIT 10;")

    sql = st.text_area("SQL Query", value=default_sql, height=160)
    c1, c2 = st.columns([1, 4])
    run    = c1.button("▶ Run Query", type="primary")
    export = c2.checkbox("Show export options", value=False)

    if run:
        sql_clean = sql.strip()
        if not sql_clean:
            st.warning("Enter a SQL query first.")
        else:
            try:
                df = pd.read_sql_query(sql_clean, conn)
                st.success(f"Returned {len(df)} rows.")
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.session_state.history.append(sql_clean)

                if export:
                    csv = df.to_csv(index=False)
                    st.download_button("⬇️ Download CSV", csv, "query_result.csv", "text/csv")
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.history.append(f"-- ERROR: {e}\n{sql_clean}")

with tab2:
    st.subheader("Database Schema")
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;", conn
    )["name"].tolist()

    for table in tables:
        with st.expander(f"📋 {table}"):
            info = pd.read_sql_query(f"PRAGMA table_info({table});", conn)
            st.dataframe(info[["name", "type", "notnull", "pk"]],
                         use_container_width=True, hide_index=True)
            count = pd.read_sql_query(f"SELECT COUNT(*) AS rows FROM {table};", conn)
            st.caption(f"Rows: {count.iloc[0, 0]}")

with tab3:
    if st.session_state.history:
        st.subheader("Query History")
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
        for i, q in enumerate(reversed(st.session_state.history[-20:]), 1):
            with st.expander(f"#{i}: {q[:60]}..."):
                st.code(q, language="sql")
    else:
        st.info("No queries run yet.")

with tab4:
    st.subheader("Natural Language → SQL Hints")
    st.caption("Describe what you want and get a SQL starting point.")
    nl = st.text_input("What do you want to query?",
                        placeholder="e.g. top 5 customers by sales, average salary by department")
    if st.button("💡 Generate Hint") and nl.strip():
        hint = nl_to_sql_hint(nl)
        if hint:
            st.code(hint, language="sql")
            if st.button("Use this query"):
                st.session_state["prefill"] = hint
        else:
            st.info("No matching hint found. Try keywords: top customers, revenue, stock, salary, orders, department...")

    st.subheader("Supported Hint Patterns")
    patterns = [
        "top N customers",
        "total / revenue / sales",
        "most popular product",
        "employee salary",
        "order status",
        "low stock / inventory",
        "customers by country",
        "average order value",
        "premium / tier customers",
        "department average salary",
    ]
    for p in patterns:
        st.markdown(f"• {p}")
