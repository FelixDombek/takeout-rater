# ADR-0001: Local web UI with Python backend (FastAPI)

**Status:** Accepted  
**Date:** 2026-03

---

## Context

The tool needs a browsing UI for 200k+ photos.  Options considered:

- Native GUI (tkinter, PyQt, wxWidgets)
- Electron / Tauri desktop app
- Local web UI served by a Python process

The user wants Windows compatibility and for the code to be almost entirely written by GitHub Copilot Agents.

---

## Decision

Use a **local web UI** served by a **FastAPI** Python backend, with **Jinja2** server-rendered templates and **HTMX** for progressive enhancement (infinite scroll, panel updates without full page reload).

---

## Consequences

**Positive:**
- No Node.js or native GUI toolkit dependency.
- FastAPI is well-documented, type-safe, and familiar to Copilot Agents.
- HTMX minimises JavaScript while keeping the UI interactive.
- Works identically on Windows, macOS, Linux.
- Easy to add REST/JSON API endpoints alongside rendered pages.

**Negative:**
- Requires a browser; not a "double-click to open" native app experience.
- HTMX has a learning curve; complex interactions may require more JS later.

---

## Alternatives considered

| Option | Rejected because |
|---|---|
| tkinter | Poor aesthetics, limited layout options, hard to paginate 200k photos |
| PyQt / wxWidgets | Large optional dep, harder for agents to generate good UI code |
| Electron | Requires Node.js, much heavier packaging |
| Tauri | Requires Rust toolchain, more complex for agents |
