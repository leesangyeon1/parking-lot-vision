"""Record hand-selected bays in original pixels, export at inference width 1280."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
# (x1, y1, x2, y2, visually occupied). Representative bays, not full-lot labels.
samples = {
    2: [(145,280,201,381,True), (202,280,276,381,True),
        (278,280,325,381,False), (327,280,391,381,True),
        (444,645,496,752,True), (497,645,548,752,False),
        (549,645,600,752,True), (601,645,651,752,False)],
    3: [(397,278,462,310,True), (463,278,526,310,True),
        (397,311,462,340,False), (463,311,526,340,True),
        (397,341,462,370,False), (463,341,526,370,False),
        (397,371,462,400,False), (463,371,526,400,True),
        (397,401,462,431,False), (463,401,526,431,True),
        (397,432,462,462,False), (463,432,526,462,False)],
    4: [(30,744,146,812,True), (30,813,146,873,False),
        (30,874,146,935,True), (30,936,146,991,False),
        (30,992,146,1055,True), (30,1056,146,1121,True),
        (438,1088,556,1144,True), (317,1088,437,1144,False),
        (438,1029,556,1086,False), (317,1029,437,1086,False)],
    5: [(365,250,419,353,True), (420,250,471,353,True),
        (472,250,522,353,False), (523,250,572,353,True),
        (573,250,623,353,True), (624,250,670,353,False),
        (370,612,420,720,False), (421,612,471,720,True),
        (472,612,522,720,False), (523,612,572,720,False)],
    6: [(57,154,155,204,True), (57,205,155,243,False),
        (57,324,155,368,True), (57,369,155,413,True),
        (295,118,392,162,True), (399,118,485,162,True),
        (295,163,392,208,True), (399,163,485,208,True)],
    7: [(1304,234,1448,292,True), (1304,293,1448,349,True),
        (1304,350,1448,406,True), (1304,407,1448,465,True),
        (1304,466,1448,526,True), (1304,527,1448,586,True),
        (1304,587,1448,645,True), (1304,646,1448,704,True)]
}
records = json.loads((OUT / 'detections.json').read_text())['images']
labels = []
for record in records:
    i = record['image']
    if i not in samples:
        continue
    scale = 1280 / record['original_shape'][1]
    slots = []
    expected = []
    for x1,y1,x2,y2,occupied in samples[i]:
        slots.append({'points': [[round(x*scale), round(y*scale)] for x,y in [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]]})
        expected.append(occupied)
    (OUT / f'{i:02d}-slots.json').write_text(json.dumps(slots, indent=2))
    labels.append(dict(image=i, source=record['source'], slot_file=f'{i:02d}-slots.json',
                       expected_occupied=expected, annotation_scope='Hand-selected visible bays; not the entire lot'))
(OUT / 'manual-labels.json').write_text(json.dumps(labels, indent=2))
