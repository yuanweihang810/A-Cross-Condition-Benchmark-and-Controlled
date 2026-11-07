# -*- coding: utf-8 -*-
from collections import Counter
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from cale_data import (CLASSES, CONDITIONS, build_condition_map,
                       scan_instances, split_instances)

root = Path("/root/autodl-tmp/IEEE/data_aligned")
cache = Path("/root/autodl-tmp/TPR_FER/cale_conditions.json")

instances = scan_instances(root)
print("instances", len(instances))
print("per class", Counter(x["class"] for x in instances))
cond = build_condition_map(instances, cache)
print("condition counts", Counter(cond.values()))

for inst in instances[:3]:
    names = sorted(inst["files"])
    print(inst["instance"], [(p.name, cond[str(p)]) for p in names[:5]])
print("original conditions", Counter(cond[str(p)] for x in instances for p in x["files"] if "_original" in p.name))

train, val, test = split_instances(instances, seed=0)
print("split instances", len(train), len(val), len(test))
print("split images", len(train)*20, len(val)*20, len(test)*20)
print("train per class", Counter(x["class"] for x in train))
