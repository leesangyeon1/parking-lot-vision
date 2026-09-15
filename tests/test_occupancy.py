import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parking_vision.occupancy import LotState, VoteBuffer, assign
from parking_vision.slots import load_slots


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must not access the network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def slots(tmp_path):
    path = tmp_path / "slots.json"
    path.write_text(json.dumps([
        {"points": [[x, 100], [x + 80, 100], [x + 80, 180], [x, 180]]}
        for x in (20, 120, 220)
    ]))
    return load_slots(path)


@pytest.mark.parametrize("boxes, expected", [
    ([[140, 120, 180, 160]], [False, True, False]),
    ([[400, 400, 420, 420]], [False, False, False]),
    ([[110, 120, 130, 160]], [False, True, False]),
    ([], [False, False, False]),
    ([[20, 100, 40, 120], [220, 100, 240, 120]], [True, False, True]),
])
def test_assign(slots, boxes, expected):
    assert assign(np.asarray(boxes, dtype=float).reshape(-1, 4), slots) == expected


@pytest.mark.parametrize("sequence, expected", [
    ([True, False, True], [True]), ([True, False, False], [False]),
    ([True, True, False, False], [False]),
])
def test_vote(sequence, expected):
    vote = VoteBuffer(3)
    for value in sequence:
        result = vote.push([value])
    assert result == expected


def test_vote_startup_tie_and_copy():
    vote = VoteBuffer(3)
    flags = [True, False]
    assert vote.push(flags) == flags
    flags[0] = False
    assert vote.push([False, True]) == [False, False]
    assert vote.push([True, True]) == [True, True]
    passthrough = VoteBuffer(1)
    assert passthrough.push([True]) == [True]
    assert passthrough.push([False]) == [False]


def test_invalid_vote():
    with pytest.raises(ValueError):
        VoteBuffer(0)
    vote = VoteBuffer(3)
    vote.push([True])
    with pytest.raises(ValueError):
        vote.push([True, False])


@pytest.mark.parametrize("flags", [[False, True, False], [True] * 3, [False] * 3])
def test_state(slots, flags):
    state = LotState.from_flags(slots, flags, 12)
    result = json.loads(json.dumps(state.to_dict()))
    assert set(result) == {"frame", "total", "empty_count", "occupied_count", "empty",
                           "occupied", "empty_locations", "slots"}
    assert state.frame == 12 and state.total == 3
    assert state.occupied == [i for i, flag in enumerate(flags) if flag]
    assert state.empty == [i for i, flag in enumerate(flags) if not flag]
    assert state.empty_count == len(state.empty)
    assert state.occupied_count == len(state.occupied)
    assert sorted(state.empty + state.occupied) == list(range(state.total))
    assert result["empty_locations"] == {str(i): list(slots[i].centroid) for i in state.empty}
    assert result["slots"] == [
        {"index": i, "occupied": flag, "centroid": list(slots[i].centroid)}
        for i, flag in enumerate(flags)
    ]


def test_state_rejects_missing_flags_or_invalid_indices(slots):
    with pytest.raises(ValueError):
        LotState.from_flags(slots, [True], 0)
    slots[0].index = 3
    with pytest.raises(ValueError):
        LotState.from_flags(slots, [True] * 3, 0)


@pytest.mark.parametrize("content", ["", "[]", '[{"points": [[0, 0], [1, 1]]}]',
                                        '{}', '[{"points": [[0, 0, 0]]}]'])
def test_load_rejects_invalid(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_slots(path)


def test_polygon_shape_centroid_and_order(tmp_path):
    entries = [{"points": [[0, 0], [10, 0], [5, 10]]},
               {"points": [[20, 0], [30, 0], [35, 5], [30, 10], [20, 10]]}]
    path = tmp_path / "slots.json"
    path.write_text(json.dumps(entries))
    original = load_slots(path)
    assert original[0].polygon.shape == (3, 1, 2)
    assert original[1].polygon.shape == (5, 1, 2)
    assert original[0].polygon.dtype == np.int32
    assert original[0].centroid == (5, 3)
    path.write_text(json.dumps(entries[::-1]))
    reordered = load_slots(path)
    for i in range(2):
        assert reordered[i].index == i
        np.testing.assert_array_equal(reordered[i].polygon, original[1 - i].polygon)
        assert reordered[i].centroid == original[1 - i].centroid
    boxes = np.array([[1, 1, 5, 3]], dtype=float)
    assert assign(boxes, reordered) == assign(boxes, original)[::-1]


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (640, 400))
    assert writer.isOpened()
    try:
        for i in range(5):
            writer.write(np.full((400, 640, 3), i * 40, dtype=np.uint8))
    finally:
        writer.release()
    return path


@pytest.mark.parametrize("every, indices", [(1, [0, 1, 2, 3, 4]), (2, [0, 2, 4])])
def test_cli_video_json_append_and_resize(tmp_path, slots, video, monkeypatch, capsys, every, indices):
    from parking_vision import main as cli

    frames = []
    def boxes(frame):
        frames.append(frame.copy())
        return np.array([[140, 120, 180, 160]], dtype=float)
    monkeypatch.setattr(cli, "Detector", lambda *args: SimpleNamespace(boxes=boxes))
    output = tmp_path / "nested" / "lot.mp4"
    json_out = tmp_path / "nested" / "state.jsonl"
    json_out.parent.mkdir()
    json_out.write_text('{"previous": true}\n')
    assert cli.main([str(video), "--slots", str(tmp_path / "slots.json"), "--width", "320",
                     "--every", str(every), "--json-out", str(json_out), "--save", str(output)]) == 0
    states = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [state["frame"] for state in states] == indices
    assert all(state["occupied"] == [1] for state in states)
    assert all(state["empty_locations"] == {"0": [60, 140], "2": [260, 140]} for state in states)
    assert len(frames) == len(indices)
    assert all(frame.shape == (200, 320, 3) for frame in frames)
    assert [json.loads(line) for line in json_out.read_text().splitlines()] == [{"previous": True}] + states
    cap = cv2.VideoCapture(str(output))
    decoded = []
    try:
        assert cap.isOpened()
        assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(12)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded.append(frame)
    finally:
        cap.release()
    assert len(decoded) == len(indices)
    assert all(frame.shape == (200, 320, 3) for frame in decoded)


def test_overlay(slots):
    from parking_vision.annotate import draw

    frame = np.zeros((220, 400, 3), dtype=np.uint8)
    state = LotState.from_flags(slots, [False, True, False], 0)
    assert draw(frame, slots, state, np.array([[140, 100, 180, 120]])) is frame
    assert frame[100, 40].tolist() == [0, 255, 0]
    assert frame[100, 140].tolist() == [0, 0, 255]
    assert frame[110, 160].tolist() == [255, 0, 255]
    assert np.any(frame[10:25, 10:300] == 255)
    assert np.any(frame[35:51, 10:150] == 255)
    assert np.any(frame[125:145, 60:75, 1] > 100)


def test_extract_frame(video, tmp_path, capsys):
    from parking_vision.tools.extract_frame import main

    output = tmp_path / "nested" / "reference.jpg"
    assert main([str(video), str(output), "--width", "320", "--frame", "3"]) == 0
    image = cv2.imread(str(output))
    assert image.shape == (200, 320, 3)
    assert 110 < image.mean() < 130
    assert capsys.readouterr().out.strip() == "(200, 320, 3)"
    with pytest.raises(ValueError, match="Cannot read frame"):
        main([str(video), str(output), "--frame", "100"])


@pytest.mark.parametrize("failure", ["detector", "inference", "writer", "quit"])
def test_cli_cleanup_and_camera_source(tmp_path, slots, monkeypatch, failure):
    from parking_vision import main as cli

    released, sources, rates = [], [], []
    def capture(source):
        sources.append(source)
        return SimpleNamespace(isOpened=lambda: True, get=lambda key: float("nan"),
                               read=lambda: (True, np.zeros((200, 320, 3), dtype=np.uint8)),
                               release=lambda: released.append("capture"))
    def fail(*args):
        raise RuntimeError("injected failure")
    detector = fail if failure == "detector" else lambda *args: SimpleNamespace(
        boxes=fail if failure == "inference" else lambda frame: np.empty((0, 4)))
    def writer(path, codec, fps, shape):
        rates.append(fps)
        return SimpleNamespace(isOpened=lambda: False, release=lambda: released.append("writer"))
    monkeypatch.setattr(cli.cv2, "VideoCapture", capture)
    monkeypatch.setattr(cli.cv2, "VideoWriter", writer)
    monkeypatch.setattr(cli.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cli.cv2, "waitKey", lambda *args: ord("q") if failure == "quit" else -1)
    monkeypatch.setattr(cli.cv2, "destroyAllWindows", lambda: released.append("windows"))
    monkeypatch.setattr(cli, "Detector", detector)
    args = ["0", "--slots", str(tmp_path / "slots.json"), "--show", "--save", str(tmp_path / "lot.mp4")]
    if failure == "quit":
        assert cli.main(args) == 0
    else:
        with pytest.raises((RuntimeError, ValueError)):
            cli.main(args)
    assert sources == [0]
    assert "capture" in released and "windows" in released
    if failure == "writer":
        assert "writer" in released and rates == [25]


@pytest.mark.parametrize("flag, value", [("--width", "0"), ("--vote", "-1"),
                                        ("--every", "0"), ("--conf", "nan")])
def test_cli_invalid_arguments(flag, value):
    from parking_vision.main import main

    with pytest.raises(SystemExit) as error:
        main(["unused.mp4", flag, value])
    assert error.value.code == 2


def test_vehicle_class_ids():
    from parking_vision.occupancy import vehicle_class_ids

    assert vehicle_class_ids({0: "plane", 1: "small vehicle", 2: "large-vehicle", 3: "ship"}) == [1, 2]
    assert vehicle_class_ids({0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}) == [2, 3, 5, 7]
    with pytest.raises(ValueError):
        vehicle_class_ids({0: "plane", 1: "ship"})


def test_detector_uses_obb_and_infers_classes(monkeypatch):
    import types
    from parking_vision.occupancy import Detector

    calls = {}
    xyxy = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array([[1, 2, 3, 4]])))

    class Fake:
        names = {0: "plane", 1: "small vehicle", 2: "large vehicle"}

        def predict(self, frame, **kwargs):
            calls.update(kwargs)
            return [SimpleNamespace(obb=SimpleNamespace(xyxy=xyxy), boxes=None)]

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=lambda path: Fake()))
    detector = Detector("fake-obb.pt", None, 0.3, "cpu", 1280)
    np.testing.assert_array_equal(detector.boxes(np.zeros((8, 8, 3), np.uint8)), [[1.0, 2.0, 3.0, 4.0]])
    assert calls["classes"] == [1, 2] and calls["imgsz"] == 1280 and calls["conf"] == 0.3


def test_detect_slots_on_synthetic_lot():
    from parking_vision.tools import detect_slots as ds

    assert ds.estimate_pitch([40, 41, 80, 39, 120]) == pytest.approx(40, abs=1)   # hidden lines = 2x, 3x gaps
    assert ds.cluster_1d([10, 11, 12, 30, 31], 3) == [11, 30.5]
    assert ds.fill_row([100, 140, 220], 40) == [100, 140, 180, 220]                # interpolates missing 180
    assert ds.fill_row([100, 140], 40, extra_points=[200]) == [100, 140, 180, 220]  # extends for a car at 200
    # synthetic top-down lot: two rows of 6 stalls (pitch 40, depth 90) sharing a spine at x=300
    frame = np.full((400, 600, 3), 60, np.uint8)
    for y in range(60, 60 + 7 * 40, 40):
        cv2.line(frame, (210, y), (390, y), (255, 255, 255), 2)
    cv2.line(frame, (300, 60), (300, 300), (255, 255, 255), 2)
    slots, angle = ds.detect_slots(frame, min_len=25, model=None)
    assert abs(angle) < 1 and len(slots) == 12
    centroids = sorted((np.mean([p[0] for p in s["points"]]), np.mean([p[1] for p in s["points"]])) for s in slots)
    assert all(abs(cx - 255) < 8 for cx, _ in centroids[:6]) and all(abs(cx - 345) < 8 for cx, _ in centroids[6:])
    assert sorted(round(cy) for _, cy in centroids[:6]) == pytest.approx([80, 120, 160, 200, 240, 280], abs=3)


def test_fit_trims_long_lists_and_terminates():
    from parking_vision.annotate import _fit

    text = "Empty slots: " + ", ".join(map(str, range(300)))
    out = _fit(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 300)
    assert out.startswith("Empty slots: 0, 1") and out.endswith(", ...")
    assert cv2.getTextSize(out, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0] <= 300
    assert _fit("Empty slots: 1, 2", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1000) == "Empty slots: 1, 2"
