"""Export manually traced visible bay rows; labels are evaluation-only, never detector input."""
import hashlib
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / 'config/examples'
DEST.mkdir(parents=True, exist_ok=True)
manifest = []


def save(key, source, groups):
    image = cv2.imread(str(source))
    h, w = image.shape[:2]
    scale = 1280 / w
    slots, truth = [], []
    for points, occupied in groups:
        slots.append({'points': [[min(1279, round(x * scale)), min(round(h * scale)-1, round(y * scale))] for x, y in points]})
        truth.append(occupied)
    (DEST / f'{key}.json').write_text(json.dumps(slots, indent=2) + '\n')
    manifest.append(dict(key=key, source=str(source.relative_to(ROOT)), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                         width=1280, shape=[round(h*scale),1280], slots=f'config/examples/{key}.json',
                         expected_occupied=truth, scope='Visible marked bays in the image, including identifiable cropped edge bays; excludes aisles, hatch zones, landscaping, and unresolvable tree cover.'))


def rows(x1, x2, edges, occupied=(), unknown=()):
    return [([(x1,y1),(x2,y1),(x2,y2),(x1,y2)], None if i in unknown else i in occupied)
            for i,(y1,y2) in enumerate(zip(edges,edges[1:]))]


def cols(y1, y2, edges, occupied=(), unknown=()):
    return [([(x1,y1),(x2,y1),(x2,y2),(x1,y2)], None if i in unknown else i in occupied)
            for i,(x1,x2) in enumerate(zip(edges,edges[1:]))]


sources = {p.stem:p for p in (ROOT/'Example photos').rglob('*.png')}
# nigh1: three double rows, west perimeter, and the visible north perimeter.
y = [278,310,341,371,401,431,462,492,523,553,584,614,645,675,706,736,767,797,828,858,889,919,950,977]
g = rows(396,462,y,[0,6,20,22]) + rows(463,528,y,[0,1,3,4,12,18,21])
g += rows(613,680,y,[0,6]) + rows(681,746,y,[3,22])
g += rows(828,896,y) + rows(897,963,y)
west=[141,169,199,230,260,290,322,353,384,414,444,475,505,535,566,597,627,658,688,719,750,780,810,841,871,902,932,962,992,1022,1052,1083]
g += rows(241,314,west,[0,1,2,3,4,5,7,8,9,10,12,13,16,18,20,23,26,27,28,29,30])
g += cols(62,132,[325,355,385,414,444,474],[0,1,2,3])
g += cols(62,132,[647,678,709,740,771,802,833,864,895,926,957,988,1019],unknown=list(range(12)))
g += cols(1100,1155,[325,355,385,414,444,474],[0,2],unknown=[1,3,4])
save('nigh1',sources['nigh1'],g)
# nigh2: three double rows, north row and side perimeter rows.
x=[336,370,421,472,522,573,624,671,720,771,821,871,922,972,1022,1071,1122,1171,1222,1272,1322,1372,1422,1472,1506]
g = cols(249,354,x,[0,1,2,4,5,9,10,19]) + cols(355,467,x,[1,2,4,17])
g += cols(609,720,x,[2]) + cols(721,827,x)
g += cols(970,1084,x) + cols(1085,1187,x)
g += cols(0,107,[119,170,222,272,323,374,425,475,526,576,627,674,725,777,824,875,926,973,1025,1075,1123,1174,1226,1275,1324,1375,1427,1474,1525,1577,1623,1675],
          [0,1,3,4,5,6,7,8,13,23],unknown=[30])
g += rows(14,113,list(range(287,1155,51))+[1187],[0,1,2,3,4,5,6,8])
g += rows(1706,1818,list(range(331,1142,54))+[1187],unknown=[0,1])
save('nigh2',sources['nigh2'],g)
# park1: crowded lot. Do not count the white vehicle in the aisle or hatch zones.
g = rows(55,156,[4,55,103,150,202,246],[0,1,2,3])
g += rows(55,156,[285,324,369,415,463,510,556,603,650,696,742,789,836,882,929,976,1023,1067],list(range(1,17)))
y=[69,114,162,207,254,300,347,394,440,486,532,579,624,671,717,765,815]
g += rows(294,392,y,list(range(16))) + rows(398,488,y,list(range(16)))
save('park1',sources['park1'],g)
# park2: perimeter around the building. Tree-obscured northwest bays are excluded.
g = cols(0,119,[268,329,388,447,506,565,626],list(range(6)))
g += cols(0,119,[627,689])  # accessible bay; adjacent striped aisle excluded
# DOTA finds vehicles; the empty accessible bay still comes from the slot map.
g += cols(0,123,[768,831,894,956,1019,1080,1143,1206,1266],list(range(8)))
g += rows(1304,1448,[232,292,350,408,466,526,586,645,704,766,826,886],list(range(11)))
for y1,y2 in zip([215,287,349,411,477,540,604,668,733,795,857],[287,349,411,477,540,604,668,733,795,857,901]):
    right1, right2 = round(174-y1*.105), round(174-y2*.105)
    g.append(([(0,y1),(right1,y1),(right2,y2),(0,y2)],True))
save('park2',sources['park2'],g)
# Sparse lot: all marked rows, including cropped but identifiable edge bays.
sparse=next(p for name,p in sources.items() if name not in {'nigh1','nigh2','park1','park2'})
g=rows(30,148,[49,106,164,221,278,336,393,451,509,567,624,682,740,812,873,936,994,1056,1120,1179],[12,14,16,17,18])
y=[123,164,222,280,338,394,452,509,567,625,683,740,798,855,913,971,1029,1087,1144,1179]
g += rows(316,437,y) + rows(438,559,y,[17])
y2=[50,109,166,222,280,339,396,455,512,568,626,685,743,800,857,915,973,1030,1088,1145,1179]
g += rows(737,855,y2) + rows(856,986,y2)
g += rows(1154,1277,y) + rows(1278,1399,y)
g += rows(1568,1689,[51,109,168,226,283,340,397,454,512,569,627,685,743,801,858,916,974,1032,1090,1148,1179],unknown=[3,4,14,15,16,17,18,19])
save('sparse',sparse,g)
(DEST/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
for r in manifest:
    print(r['key'],len(r['expected_occupied']),'bays;',sum(v is True for v in r['expected_occupied']),'visually occupied;',sum(v is None for v in r['expected_occupied']),'uncertain')
