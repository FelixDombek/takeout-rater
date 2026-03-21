# ADR-0004: Library state location (sibling `takeout-rater/` directory)

**Status:** Accepted  
**Date:** 2026-03

---

## Context

The app needs somewhere to store:
- `library.sqlite` (metadata + scores)
- `thumbs/` (thumbnail cache)
- `cache/` (scorer intermediates)
- `logs/`
- `exports/`

Constraints:
- The Takeout archive is **immutable** — the app must never modify it.
- The user has 715 GB of takeout data; state may also be large (thumbnails, scores).
- Windows-friendly paths.
- Easy to find / back up together with the takeout.

---

## Decision

Store all mutable app state in a **sibling directory** called `takeout-rater/` next to the `Takeout/` folder:

```
<library-root>/
  Takeout/           ← Google Photos archive (never modified)
  takeout-rater/     ← created by the app on first run
    library.sqlite
    thumbs/
    cache/
    logs/
    exports/
```

The user points the app at `<library-root>` (the directory *containing* `Takeout/`).

---

## Consequences

**Positive:**
- Takeout is truly immutable; the app only reads from it.
- App state lives next to the data it describes — easy to find, move, and back up.
- No per-user config directory needed for the common case.
- Multiple separate takeouts can each have their own `takeout-rater/` alongside.

**Negative:**
- If the takeout lives on a read-only mount, a `--data-dir` override will be needed.
- The `takeout-rater/` directory appears inside the user's takeout storage location.

---

## Alternatives considered

| Option | Rejected because |
|---|---|
| Inside `Takeout/` | Modifies the archive directory |
| `%LOCALAPPDATA%/takeout-rater/` | Hard to associate state with a specific takeout; clutters app data |
| User-configurable only | Too much friction for the common case |
