import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parking_vision.occupancy import Detector, LotState, VoteBuffer, assign
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
    assert np.any(frame[130:150, 50:70, 1] > 100)


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
                                        ("--every", "0"), ("--imgsz", "0"), ("--conf", "nan")])
def test_cli_invalid_arguments(flag, value):
    from parking_vision.main import main

    with pytest.raises(SystemExit) as error:
        main(["unused.mp4", "--slots", "unused.json", flag, value])
    assert error.value.code == 2


@pytest.mark.parametrize("names, task, expected", [
    ({0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}, "detect", [2, 3, 5, 7]),
    ({0: "plane", 4: "tennis court", 9: "large vehicle", 10: "small vehicle"}, "obb", [9, 10]),
    ({4: "small-vehicle", 5: "large-vehicle"}, "obb", [4, 5]),
])
def test_detector_class_mapping(monkeypatch, names, task, expected):
    # Exercise initialization on a stand-in; no Detector or real YOLO model is instantiated.
    fake_model = SimpleNamespace(names=names, task=task)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda model: fake_model))
    receiver = SimpleNamespace()
    Detector.__init__(receiver, "fake.pt", None, .25, "cpu")
    assert receiver.classes == expected
    assert receiver.imgsz is None
    Detector.__init__(receiver, "fake.pt", [next(iter(names))], .25, "cpu", 640)
    assert receiver.classes == [next(iter(names))] and receiver.imgsz == 640


def test_detector_rejects_unknown_classes(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(
        YOLO=lambda model: SimpleNamespace(names={0: "object"}, task="detect")))
    with pytest.raises(ValueError, match="--classes"):
        Detector.__init__(SimpleNamespace(), "fake.pt", None, .25, "cpu")


@pytest.mark.parametrize("oriented", [False, True])
@pytest.mark.parametrize("coordinates", [[], [[10., 20., 30., 40.]]])
def test_detector_box_results(oriented, coordinates):
    array = np.array(coordinates, dtype=float).reshape(-1, 4)
    tensor = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: array))
    detections = SimpleNamespace(xyxy=tensor)
    calls = []
    def predict(frame, **kwargs):
        calls.append(kwargs)
        return [SimpleNamespace(obb=detections if oriented else None,
                                boxes=None if oriented else detections)]
    receiver = SimpleNamespace(model=SimpleNamespace(predict=predict), classes=[9, 10],
                               conf=.25, device="cpu", imgsz=1280)
    result = Detector.boxes(receiver, np.zeros((100, 200, 3), dtype=np.uint8))
    np.testing.assert_array_equal(result, array)
    assert result.shape == (len(coordinates), 4)
    assert calls == [dict(classes=[9, 10], conf=.25, device="cpu", imgsz=1280, verbose=False)]


def test_cli_requires_explicit_slot_map():
    from parking_vision.main import main
    with pytest.raises(SystemExit) as error:
        main(["any.mp4"])
    assert error.value.code == 2


def test_cli_saves_still_image_and_checks_coordinates(tmp_path, slots, monkeypatch, capsys):
    from parking_vision import main as cli
    source = tmp_path / "photo.png"
    assert cv2.imwrite(str(source), np.zeros((200, 320, 3), dtype=np.uint8))
    monkeypatch.setattr(cli, "Detector", lambda *args: SimpleNamespace(
        boxes=lambda frame: np.array([[140.,120.,180.,160.]])))
    output = tmp_path / "nested" / "overlay.png"
    args = [str(source), "--slots", str(tmp_path / "slots.json"), "--width", "320", "--save", str(output)]
    assert cli.main(args) == 0
    state = json.loads(capsys.readouterr().out)
    assert state["total"] == 3 and state["occupied"] == [1]
    assert cv2.imread(str(output)).shape == (200, 320, 3)
    with pytest.raises(ValueError, match="Slot coordinates exceed"):
        cli.main([str(source), "--slots", str(tmp_path / "slots.json"), "--width", "100"])


@pytest.mark.parametrize("shape, expected_size", [((100, 200, 3), 1280), ((200, 100, 3), 1920)])
def test_detector_auto_resolution(shape, expected_size):
    sizes = []
    tensor = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.empty((0, 4))))
    def predict(frame, **kwargs):
        sizes.append(kwargs["imgsz"])
        return [SimpleNamespace(obb=SimpleNamespace(xyxy=tensor), boxes=None)]
    receiver = SimpleNamespace(model=SimpleNamespace(predict=predict), classes=[9, 10],
                               conf=.25, device="cpu", imgsz=None)
    Detector.boxes(receiver, np.zeros(shape, dtype=np.uint8))
    assert sizes == [expected_size]


def test_example_maps_match_sources_and_cover_known_bays():
    import hashlib

    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "config/examples/manifest.json").read_text())
    expected_totals = {"nigh1": 191, "nigh2": 209, "park1": 54, "park2": 37, "sparse": 155}
    for item in manifest:
        source = root / item["source"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == item["sha256"]
        slots = load_slots(root / item["slots"])
        assert len(slots) == expected_totals[item["key"]] == len(item["expected_occupied"])
        height, width = item["shape"]
        for slot in slots:
            assert (slot.polygon >= 0).all()
            assert (slot.polygon[:, 0] < (width, height)).all()
        # Conservation still holds for hundreds of slots, independent of model predictions.
        state = LotState.from_flags(slots, [flag is True for flag in item["expected_occupied"]], 0)
        assert state.empty_count + state.occupied_count == len(slots)
