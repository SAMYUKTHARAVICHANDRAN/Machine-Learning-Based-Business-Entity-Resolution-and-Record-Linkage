"""Deterministic fit/dev/holdout split of train S1 by hashed entity id (80/10/10).
Dev and holdout samples exclude the 30k entities the v1 model was trained on."""
import sys, random, hashlib, pickle
sys.path.insert(0, "."); from lib_data import load_tuples, load_gt
D = "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
s1 = load_tuples(f"{D}/train_source1.tsv")
gt = load_gt(f"{D}/train_ground_truth.tsv")

# reproduce v1's training sample exactly (run_train.py: seed 42 shuffle, first 30k of 85%)
v1 = list(s1); random.seed(42); random.shuffle(v1)
v1_train = {r[0] for r in v1[:int(0.85 * len(v1))][:30000]}

def bucket(eid):
    return int.from_bytes(hashlib.blake2b(eid.encode(), digest_size=4).digest(), "little") % 100

split = {r[0]: ("fit" if bucket(r[0]) < 80 else "dev" if bucket(r[0]) < 90 else "hold") for r in s1}
rng = random.Random(7)
dev_all  = [r[0] for r in s1 if split[r[0]] == "dev"  and r[0] not in v1_train]
hold_all = [r[0] for r in s1 if split[r[0]] == "hold" and r[0] not in v1_train]
fit_all  = [r[0] for r in s1 if split[r[0]] == "fit"]
dev, hold = rng.sample(dev_all, 20000), rng.sample(hold_all, 20000)
pickle.dump({"split": split, "dev": dev, "hold": hold, "fit": fit_all, "v1_train": v1_train},
            open("work/splits.pkl", "wb"))
for name, ids in (("fit", fit_all), ("dev sample", dev), ("holdout sample", hold)):
    sing = sum(1 for i in ids if not gt[i]) / len(ids)
    print(f"{name:<15} {len(ids):>9,}   singletons {sing:.2%}")
print(f"v1 training entities excluded from dev/holdout: {len(v1_train):,}")
