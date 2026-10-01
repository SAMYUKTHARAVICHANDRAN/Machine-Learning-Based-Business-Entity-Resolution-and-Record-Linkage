"""Fast vectorized data loading. tuples via zip, not iterrows()."""
import pandas as pd

DATA = "6ab10eb3b23ba_student_resource/student_resource/dataset"

def load_tuples(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return list(zip(df["entity_id"], df["business_name"],
                    df["business_address"], df["country"]))

def load_gt(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    out = {}
    for sid, matched in zip(df["source1_entity_id"], df["matched_entity_ids"]):
        out[sid] = {x.strip() for x in matched.split(",") if x.strip()} if matched else set()
    return out
