# Todo App

> A web-based task management application built with Flask and SQLAlchemy.

## Overview

A full CRUD (Create, Read, Update, Delete) todo web application using Flask as the web framework and Flask-SQLAlchemy with SQLite as the database backend. The frontend uses Bootstrap 4 for styling and Jinja2 templates for rendering.

## Features

- **Add tasks** via a form on the main page
- **View all tasks** in a table sorted by creation date
- **Update tasks** with inline editing (pre-populated form)
- **Delete tasks** with a single click
- Each task stores content, completion status, and publication date
- Bootstrap 4 responsive UI
- SQLite database (`test.db`) for persistent storage

## Project Structure

```
Todo_app/
├── app.py
├── requirements.txt
├── test.db
├── static/
│   └── css/
│       └── style.css
├── templates/
│   ├── base.html
│   └── index.html
└── Readme.md
```

## Requirements

- Python 3.x
- Flask 1.1.2
- Flask-SQLAlchemy 2.4.4

## Installation

```bash
cd "Todo_app"
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

The app starts a Flask development server at `http://127.0.0.1:5000/` with debug mode enabled.

### Routes

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Display all tasks |
| POST | `/` | Add a new task |
| GET | `/delete/<id>` | Delete a task by ID |
| GET/POST | `/update/<id>` | Update a task by ID |

## How It Works

1. **Database Model (`Todo`):** Stores `id` (primary key), `content` (string, max 200 chars), `completed` (integer, default 0), and `pub_date` (datetime, defaults to `datetime.utcnow`).
2. **Index route:** On GET, queries all tasks ordered by `pub_date` and renders `index.html`. On POST, creates a new `Todo` with the submitted content.
3. **Delete route:** Fetches the task by ID (returns 404 if not found), deletes it from the database.
4. **Update route:** On GET, passes the task to the template for pre-populating the form. On POST, updates `task.content` and commits.
5. **Templates:** `base.html` provides the Bootstrap layout; `index.html` extends it with the task table and form.

## Configuration

- **Database URI:** Hardcoded as `sqlite:///test.db` in `app.py`.
- **Debug mode:** Enabled (`app.run(debug=True)`).
- **SQLAlchemy track modifications:** Disabled.

## Limitations

- The `completed` field exists in the model but is never used in the UI or routes.
- Delete uses a GET request instead of POST/DELETE, which is not RESTful and could be triggered by web crawlers.
- Generic error messages ("There is an issue") with bare `except` clauses — no logging or specific error handling.
- No input validation or sanitization on task content.
- Debug mode is hardcoded to `True` (should not be used in production).

## Security Notes

- Debug mode is enabled, which exposes the Werkzeug debugger in production if not changed.
- No CSRF protection on forms.

## License

Not specified.
