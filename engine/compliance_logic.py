"""Rule-based PPE compliance from detections + spatial overlap.

Status: SAFE (green) | ATTENTION (yellow, minor) | DANGER (red, critical).
Head protection is a critical control (no helmet = DANGER); vest is secondary
(ATTENTION). A vest violation is raised only when the model is confident
(NO_VEST_CONF) that a worker has NO vest, so a model miss does not cause a
false alarm. Set strict=True to flag any worker without a detected vest.
"""
from dataclasses import dataclass, field

HELMETS = {"white", "blue", "red", "yellow", "hat", "helmet", "hardhat"}
NO_HELMET = {"head", "nohelmet", "no-hardhat"}
VESTS = {"vest", "safety vest"}
NO_VEST = {"novest", "no-safety vest"}
EYE = {"glass"}
PERSON = {"person", "worker"}

NO_VEST_CONF = 0.6   # min confidence to trust a "NO vest" detection


@dataclass
class WorkerStatus:
    box: tuple
    status: str
    violations: list = field(default_factory=list)


def _inter_over_area(inner, outer):
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def evaluate(dets, require_glasses=False, min_overlap=0.5, strict=False, no_vest_conf=NO_VEST_CONF):
    """dets: iterable of objects with .cls_name .conf .box. Returns list[WorkerStatus]."""
    dets = [(d.cls_name.lower(), d.box, d.conf) for d in dets]
    persons = [b for n, b, _ in dets if n in PERSON]
    results = []
    for p in persons:
        ph = p[3] - p[1]
        upper = (p[0], p[1], p[2], p[1] + 0.35 * ph)   # head region

        def inside(names, region, min_conf=0.0):
            return any(n in names and c >= min_conf
                       and _inter_over_area(b, region) >= min_overlap
                       and _inter_over_area(b, p) >= min_overlap for n, b, c in dets)

        violations = []
        if not inside(HELMETS, upper) or inside(NO_HELMET, upper):
            violations.append("NO_HELMET")
        has_vest = inside(VESTS, p)
        sure_no_vest = inside(NO_VEST, p, no_vest_conf)
        if (strict and not has_vest) or (sure_no_vest and not has_vest):
            violations.append("NO_VEST")
        if require_glasses and not inside(EYE, upper):
            violations.append("NO_EYE_PROTECTION")

        if "NO_HELMET" in violations:
            status = "DANGER"
        elif violations:
            status = "ATTENTION"
        else:
            status = "SAFE"
        results.append(WorkerStatus(p, status, violations))
    return results


def overall_status(workers):
    if any(w.status == "DANGER" for w in workers): return "DANGER"
    if any(w.status == "ATTENTION" for w in workers): return "ATTENTION"
    return "SAFE"


def compliance_rate(workers):
    return 100.0 * sum(w.status == "SAFE" for w in workers) / len(workers) if workers else 100.0
