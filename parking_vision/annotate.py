import cv2

GREEN, RED, MAGENTA, DARK, WHITE = (0, 255, 0), (0, 0, 255), (255, 0, 255), (25, 25, 25), (255, 255, 255)


def _fit(text, font, scale, max_width):
    """Trim a comma list from the right until it fits max_width px."""
    items = text.split(", ")
    while len(items) > 1 and cv2.getTextSize(", ".join(items), font, scale, 1)[0][0] > max_width:
        items = items[:-2] + ["..."] if items[-1] == "..." else items[:-1] + ["..."]
    return ", ".join(items)


def draw(frame, slots, state, boxes):
    font = cv2.FONT_HERSHEY_SIMPLEX
    s = min(1.6, 0.6 * frame.shape[1] / 400)  # text scale grows with frame width
    # translucent fill so empty (green) vs occupied (red) reads at a glance
    fill = frame.copy()
    for slot in slots:
        cv2.fillPoly(fill, [slot.polygon], RED if slot.index in state.occupied else GREEN)
    cv2.addWeighted(fill, 0.25, frame, 0.75, 0, frame)
    for slot in slots:
        color = RED if slot.index in state.occupied else GREEN
        cv2.polylines(frame, [slot.polygon], True, color, 2)
        text, (x, y) = str(slot.index), slot.centroid
        side = min(cv2.boundingRect(slot.polygon)[2:])  # label must fit inside its own slot
        ls = min(s, 0.5 * side / 22)
        (w, h), baseline = cv2.getTextSize(text, font, ls, 1)
        while w > 0.9 * side and ls > 0.3:  # shrink to fit the slot width
            ls *= 0.85
            (w, h), baseline = cv2.getTextSize(text, font, ls, 1)
        x, y = x - w // 2, y + h // 2  # center label on the slot
        cv2.rectangle(frame, (x - 2, y - h - 2), (x + w + 2, y + baseline + 2), DARK, -1)
        cv2.putText(frame, text, (x, y), font, ls, color, 1 if ls < 0.8 else 2, cv2.LINE_AA)
    for x1, y1, x2, y2 in boxes:
        cv2.circle(frame, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 3, MAGENTA, -1)
    lines = [f"Total {state.total} | Empty {state.empty_count} | Occupied {state.occupied_count}",
             _fit(f"Empty slots: {', '.join(map(str, state.empty))}", font, s, frame.shape[1] - 20),
             _fit(f"Occupied slots: {', '.join(map(str, state.occupied))}", font, s, frame.shape[1] - 20)]
    for i, text in enumerate(lines):
        y = int((24 + 26 * i) * s / 0.6)
        (w, h), baseline = cv2.getTextSize(text, font, s, 1)
        cv2.rectangle(frame, (6, y - h - 3), (12 + w, y + baseline + 3), DARK, -1)
        cv2.putText(frame, text, (10, y), font, s, WHITE, 1, cv2.LINE_AA)
    return frame
