import json
from types import SimpleNamespace as D
from data.dataset_pipeline import voc_to_yolo, coco_to_yolo, oversample_minority
from engine.compliance_logic import evaluate, overall_status

PERSON = (100, 100, 200, 400)


def d(n, box): return D(cls_name=n, conf=.9, box=box)


def test_safe():
    w = evaluate([d("person", PERSON), d("white", (120, 100, 180, 140)), d("vest", (110, 190, 190, 300))])
    assert overall_status(w) == "SAFE"


def test_no_helmet_is_danger():
    w = evaluate([d("person", PERSON), d("head", (120, 100, 180, 140)), d("vest", (110, 190, 190, 300))])
    assert w[0].status == "DANGER" and "NO_HELMET" in w[0].violations


def test_no_vest_is_attention():
    w = evaluate([d("person", PERSON), d("blue", (120, 100, 180, 140)), d("NO-Safety Vest", (110, 190, 190, 300))])
    assert w[0].status == "ATTENTION" and w[0].violations == ["NO_VEST"]
    assert evaluate([d("person", PERSON), d("blue", (120, 100, 180, 140))], strict=True)[0].violations == ["NO_VEST"]


def test_voc(tmp_path):
    x = tmp_path / "a.xml"
    x.write_text("<annotation><size><width>100</width><height>100</height></size><object><name>Hat</name>"
                 "<bndbox><xmin>10</xmin><ymin>10</ymin><xmax>30</xmax><ymax>50</ymax></bndbox></object></annotation>")
    assert voc_to_yolo(x, ["Worker", "Hat"])[0].startswith("1 0.2")


def test_coco(tmp_path):
    j = tmp_path / "c.json"
    j.write_text(json.dumps({"images": [{"id": 1, "file_name": "i.jpg", "width": 100, "height": 100}],
                             "categories": [{"id": 1, "name": "person"}],
                             "annotations": [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 50, 50]}]}))
    coco_to_yolo(j, tmp_path / "out")
    assert (tmp_path / "out" / "i.txt").read_text().startswith("0 0.25")


def test_oversample(tmp_path):
    (tmp_path / "a.txt").write_text("0 .5 .5 .1 .1\n" * 10)
    (tmp_path / "b.txt").write_text("1 .5 .5 .1 .1")
    assert oversample_minority(tmp_path)


def test_css_labels():
    ok = evaluate([d("Person", PERSON), d("Hardhat", (120, 100, 180, 140)), d("Safety Vest", (110, 190, 190, 300))])
    assert ok[0].status == "SAFE"
    bad = evaluate([d("Person", PERSON), d("NO-Hardhat", (120, 100, 180, 140)), d("Safety Vest", (110, 190, 190, 300))])
    assert bad[0].status == "DANGER"
    nov = evaluate([d("Person", PERSON), d("Hardhat", (120, 100, 180, 140)), d("NO-Safety Vest", (110, 190, 190, 300))])
    assert nov[0].status == "ATTENTION"


def test_tall_vest_box_still_counts():
    # vest box taller than the 'torso band' (close-up worker) must still count as a vest
    w = evaluate([d("person", PERSON), d("Hardhat", (120, 100, 180, 140)), d("Safety Vest", (105, 140, 195, 395))])
    assert w[0].status == "SAFE"


def test_low_confidence_no_vest_is_ignored():
    low = D(cls_name="NO-Safety Vest", conf=0.48, box=(110, 190, 190, 300))
    w = evaluate([d("person", PERSON), d("Hardhat", (120, 100, 180, 140)), low])
    assert w[0].status == "SAFE"
