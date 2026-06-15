# Dropdown alignment

Alignment convention for dropdown/popover menus opened from a trigger in a horizontal bar (the tabs row, panel headers). The extensions menu is the reference implementation.

## Convention

- The popup is `position: absolute; top: 100%` on a `position: relative` trigger wrapper, so it hangs directly below the trigger.
- Anchor the popup to the **same horizontal edge the trigger sits against in its row**:
  - Trigger on the left side of the row → `left: 0` (popup grows rightward, reading left to right under the trigger). This is the default.
  - Trigger pinned to the far right of the row → `right: 0` (popup grows leftward, staying on screen).
- Match the anchor to the trigger's side. A left-positioned trigger with `right: 0` makes the box expand off to the left, away from the trigger: the wrong direction.

## Reference: extensions menu

- CSS: `.ext-menu` (relative wrapper), `.ext-trigger` (the "Extensions ▾" trigger), `.ext-pop` (the popup), `.ext-library` (standalone link to the trigger's left) in [index.css:62-70](../../../veritate_mri/web/index.css#L62-L70).
- Markup + wiring: `buildMenu()` in the inline IIFE in [index.html:2053](../../../veritate_mri/web/index.html#L2053). The library link and the dropdown are appended into `.tabs`; the trigger sits at the left end of the tabs row, so `.ext-pop` uses `left: 0`.

## Dependencies

- Trigger wrapper must be `position: relative`; the popup is `position: absolute` relative to it.

## Pitfalls

- `right: 0` on a left-aligned trigger sends the popup the wrong way. Re-anchor, do not patch with negative margins.
- The popup needs a `z-index` above sibling content (`.ext-pop` uses `50`).
