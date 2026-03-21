# ADR-0005: v1 filtering and sorting without a query language

**Status:** Accepted  
**Date:** 2026-03

---

## Context

The user wants to filter photos by scorer outputs:
> "laion_aesthetic >= 9 AND nsfw < 5 AND …"

A full query DSL (custom grammar, parser, compiler to SQL) is expressive but complex to implement and test correctly.

In v1, the primary workflows are:
- "Show me the highest-scored photos" (sort by one metric)
- "Exclude likely NSFW" (threshold on one metric)
- "Only show cluster representatives" (boolean toggle)
- "Filter by album or year" (set membership)

---

## Decision

Use **structured filter objects** combined with **AND semantics** — no query language in v1.

### Filter types

| Type | Example |
|---|---|
| `NumericRangeFilter` | `aesthetic` between 7.0 and 10.0 |
| `BooleanFilter` | `is_cluster_representative = true` |
| `SetMembershipFilter` | `album in {"Paris 2024", "Family"}` |

### Sort

- One **primary sort** field (scorer metric or metadata column) + direction.
- Optional **secondary sort** for tie-breaking (e.g. `taken_at DESC`).

### View presets

Filter + sort combinations are serialised as JSON and stored in the DB as named "view presets".  The schema is:

```json
{
  "sort": {"field": "aesthetic", "direction": "desc"},
  "filters": [
    {"type": "numeric_range", "field": "aesthetic", "min": 7.0},
    {"type": "boolean", "field": "is_cluster_representative", "value": true}
  ],
  "exclude_unscored": true
}
```

### Upgrade path

The JSON preset schema is designed so it can later be compiled from a DSL input without changing the DB schema.

---

## Consequences

**Positive:**
- Simple to implement correctly.
- Easy for agents to extend (add a new filter type = add a new class + one SQL snippet).
- UI maps directly to the data model (no DSL parsing in the browser).

**Negative:**
- No OR conditions across filters in v1.
- No computed-field filters (e.g. `score_a - score_b > 3`).
- Multi-select for set membership covers most real OR use cases (e.g. "album in A or B").

---

## Alternatives considered

| Option | Rejected because |
|---|---|
| Custom DSL + parser | Over-engineered for v1; hard to get right; deferred to later iteration |
| Raw SQL WHERE clause | Security risk (SQL injection); no UI mapping; brittle |
| GraphQL filter input | Heavy dependency; same expressiveness as structured objects |
