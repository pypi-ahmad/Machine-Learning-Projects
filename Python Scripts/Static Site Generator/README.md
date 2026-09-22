# Static Site Generator

A standard-library static-site generator that turns Markdown files into HTML,
creates a post index, copies top-level assets, and can serve the generated
directory locally.

## Start a site

```powershell
uv run python main.py init my-site
uv run python main.py build --src my-site/content --out my-site/public
```

`init` creates a site directory containing `content/hello-world.md`, an
`assets/` directory, and `ssg_config.json`. It refuses to overwrite an existing
directory.

## Add content and build

From a generated site directory:

```powershell
uv run python ..\main.py new post "My First Post"
uv run python ..\main.py build
uv run python ..\main.py serve --port 8000
```

`build` reads Markdown from the configured source directory and writes HTML,
an index page, and CSS to the configured output directory. `serve` starts a
local HTTP server and runs until interrupted with `Ctrl+C`.

## Supported Markdown

The built-in renderer supports headings, paragraphs, emphasis, inline code,
links, images, lists, blockquotes, horizontal rules, and fenced code blocks.
Front matter is a simple `key: value` block between opening and closing `---`
lines; it is not a full YAML parser.

## Limitations

Content is inserted into templates without HTML escaping. Treat source Markdown
and front matter as trusted author-controlled input. The renderer is deliberately
small and does not implement the full CommonMark specification.

## Dependencies

The project uses only Python's standard library. uv records the Python
requirement and provides the reproducible environment.
