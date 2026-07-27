# MODULE_NAME

## Files

- `ds4_MODULE.c` — implementation
- `ds4_MODULE.h` — public interface (if separate)

## Purpose

One-sentence: what this module does and why it exists as a separate compilation unit.

## Dependencies

- **Imports from**: [list of other modules/types]
- **Exports to**: [list of callers]
- **Init order**: [when in startup sequence this module activates]

## Key Types

| Type | Role |
|---|---|
| `type_name` | brief description |
| ... | |

## API Surface

### Creation / Teardown

- `fn()` — what it does, preconditions, ownership

### Core Operations

- `fn()` — what it does, when it blocks, error contract

### Query / Introspection

- `fn()` — what it returns

## Data Flow

```
[input] → [module processing] → [output]
```

Describe the shape and lifetime of the main data that passes through this module.

## Invariants

- Things that must be true for this module to work correctly.
- Memory ownership rules.
- Thread safety constraints.

## Configuration

- Environment variables read: `DS4_*`
- Compile-time flags: `#ifdef DS4_*`

## Notes

Implementation quirks, historical context, known limitations.
