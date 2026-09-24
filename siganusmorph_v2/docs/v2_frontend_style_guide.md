# V2.0 Frontend Style Guide

## Design Direction

The user-facing app follows an Apple-inspired style: quiet background, large white cards, generous spacing, minimal shadows, clear typography and blue primary actions.

## Tokens

- background: `#f5f5f7`
- card: `#ffffff`
- text: `#1d1d1f`
- secondary: `#707070`
- primary CTA: `#0071e3`
- link: `#0066cc`
- input: `#e8e8ed`
- card radius: `28px`
- button radius: `36px`

## Principles

- Do not show debug/audit/experimental labels in the user app.
- Do not show local absolute paths.
- Do not show Python tracebacks to ordinary users.
- Calibration failure should be a user-friendly message.
- Keep advanced review and model tools in the Admin Console.

## Measurement Workbench

- Use warped image as the only measurement background.
- Draw small visible points with a larger invisible hit radius.
- Show hover labels, crosshair, magnifier and keyboard nudge.
- P7V is a derived point and is not directly dragged by default.
- Local dragging updates temporary overlay and measurement preview.
- Apply changes calls `/api/update-points`.

