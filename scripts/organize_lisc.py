from pathlib import Path
import json
import os
import pandas as pd
from PIL import Image

root = Path("./data/LISC")

idx2label = {1: 'Neutrophil',
             2: 'Lymphocyte',
             3: 'Monocyte',
             4: 'Eosinophil',
             5: 'Basophil'
             }

train = json.load(open((root/"Train.json")))
test = json.load(open((root/"Test.json")))

df = []
for filename, idx in train.items():
    path = os.path.join("Train", filename)
    label = idx2label[idx]
    df.append({"path": path, "label": label})
for filename, idx in test.items():
    path = os.path.join("Test", filename)
    label = idx2label[idx]
    df.append({"path": path, "label": label})
df = pd.DataFrame(df)

LISCCropped = Path(os.path.join('./data/', "LISCCropped"))
LISCCropped.mkdir(exist_ok=True)
for i, row in df.iterrows():
    label = row["label"]
    pathfull = Path(os.path.join(root, row["path"]))
    img_root = Path(os.path.join(LISCCropped, label))
    img_root.mkdir(exist_ok=True)
    img = Image.open(pathfull)
    img.save((img_root / pathfull.with_suffix(".jpg").name))
