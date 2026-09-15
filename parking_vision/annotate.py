import cv2


def draw(frame, slots, state, boxes):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for slot in slots:
        color = (0, 0, 255) if slot.index in state.occupied else (0, 255, 0)
        cv2.polylines(frame, [slot.polygon], True, color, 2)
        text, (x, y) = str(slot.index), slot.centroid
        (w, h), baseline = cv2.getTextSize(text, font, 0.4, 1)
        x, y = x - w // 2, y + h // 2
        cv2.rectangle(frame, (x - 2, y - h - 2), (x + w + 2, y + baseline + 2), (25, 25, 25), -1)
        cv2.putText(frame, text, (x, y), font, 0.4, color, 1, cv2.LINE_AA)
    for x1, y1, x2, y2 in boxes:
        cv2.circle(frame, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 3, (255, 0, 255), -1)
    lines = [f"Total {state.total} | Empty {state.empty_count} | Occupied {state.occupied_count}",
             f"Empty: {state.empty[:12]}" + (f" (+{len(state.empty) - 12} more; see JSON)" if len(state.empty) > 12 else "")]
    for y, text in zip((24, 50), lines):
        (w, h), baseline = cv2.getTextSize(text, font, 0.6, 1)
        cv2.rectangle(frame, (6, y - h - 3), (12 + w, y + baseline + 3), (25, 25, 25), -1)
        cv2.putText(frame, text, (10, y), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame
