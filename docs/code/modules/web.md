# Web UI

## Files

- `ds4_web.c` — embedded web interface
- `ds4_web.h` — web UI API

## Purpose

Lightweight web interface for model interaction. Serves HTML/JS chat UI from the server process.

## UI Components

Embedded HTTP handler inside ds4-server. Serves chat HTML, JS, and CSS assets. Routes map:

| Route | Content |
|---|---|
| `/` | chat UI |
| `/static/*` | JS/CSS assets |
| `/api/*` | proxied to server API |

## API Integration

All UI interactions route through server API endpoints. No direct backend access from the client.

## Configuration

Static assets compiled into the binary or loaded from disk at startup. Server runs without the web UI enabled.

## Invariants

- Web UI optional; server runs without it.
- Static assets versioned for cache busting.
- UI communicates exclusively through server API endpoints.

## See Also

- [Engine API](../engine/engine-api.md)
- [Server](server.md)

[← Back to Index](../README.md)
