import os, re, time
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image, ExifTags
from arcgis.gis import GIS
from arcgis.features import FeatureLayerCollection, Feature
from arcgis.geometry import Point

# --- config ---
BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
MODEL_PATH = BASE / "model" / "model.tflite"
IMAGES_DIR  = BASE / "images"

PORTAL = "https://aoslcps.maps.arcgis.com"
CLIENT_ID = "CpOBbpWXL80dumtM"
LAYER_TITLE = "TrailSense Conditions"
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic"}

BACKBONE = "efficientnetv2"          
APPLY_EXIF_TRANSPOSE = False        

TRAIN_HEADS = ["incision", "muddiness", "roots", "braiding"]
HEAD_TO_FIELD = {"incision":"erosion","muddiness":"mud","roots":"roots","braiding":"braiding"}
FIELDS = ["erosion","mud","roots","braiding"]

GPS_TAG = next(k for k, v in ExifTags.TAGS.items() if v == "GPSInfo")

# --- gps (minimal) ---
def _rat(x):
    try: return float(x)
    except Exception:
        try: return float(x[0]) / float(x[1])
        except Exception: return None

def _dms(dms, ref):
    d,m,s = _rat(dms[0]), _rat(dms[1]), _rat(dms[2])
    if None in (d,m,s): return None
    v = d + m/60.0 + s/3600.0
    return round(-v if ref in ("S","W") else v, 7)

def gps(p: Path):
    try:
        ex = Image.open(p)._getexif() or {}
        g = ex.get(GPS_TAG)
        if not g: return None, None
        return _dms(g[2], g[1]), _dms(g[4], g[3])  # lat, lon
    except Exception:
        return None, None

# --- arcgis: get or create ---
def get_layer(gis: GIS):
    me = gis.users.me.username
    items = gis.content.search(
        query=f'title:"{LAYER_TITLE}" AND owner:{me}',
        item_type="Feature Layer",
        max_items=5,
    )
    if items:
        print("Layer found:", items[0].title)
        return items[0].layers[0]

    print("Layer not found — creating…")
    svc = re.sub(r"[^A-Za-z0-9_]+", "_", LAYER_TITLE).strip("_")[:40]
    svc = f"{svc}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    item = gis.content.create_service(
        name=svc,
        service_type="featureService",
        create_params={"name": svc, "capabilities": "Query,Create,Update,Delete,Editing"},
    )
    item.update(item_properties={"title": LAYER_TITLE})

    flc = FeatureLayerCollection.fromitem(item)
    flc.manager.add_to_definition({
        "layers": [{
            "id": 0,
            "name": "observations",
            "type": "Feature Layer",
            "geometryType": "esriGeometryPoint",
            "objectIdField": "OBJECTID",
            "fields": [
                {"name":"OBJECTID","alias":"OBJECTID","type":"esriFieldTypeOID"},
                {"name":"image_file","alias":"image_file","type":"esriFieldTypeString","length":255},
                {"name":"photo_date","alias":"photo_date","type":"esriFieldTypeDate"},
                *[{"name":f,"alias":f,"type":"esriFieldTypeDouble"} for f in FIELDS],
            ],
        }]
    })

    # re-fetch until layers are visible (avoids needing _refresh())
    for _ in range(10):
        item2 = gis.content.get(item.id)
        try:
            lyr = item2.layers[0]
            print("Created layer:", item2.title)
            return lyr
        except Exception:
            time.sleep(1)

    raise RuntimeError("Created service but layer not available yet (retry).")

# --- model: correct TF preprocessing + output mapping ---
class M:
    def __init__(self, model_path: Path):
        import tensorflow as tf
        self.tf = tf
        self.t = tf.lite.Interpreter(model_path=str(model_path))
        self.t.allocate_tensors()
        self.inp = self.t.get_input_details()[0]
        self.out = self.t.get_output_details()
        _, h, w, _ = self.inp["shape"]
        self.h, self.w = int(h), int(w)

        if BACKBONE == "efficientnetv2":
            self.prep = tf.keras.applications.efficientnet_v2.preprocess_input
        elif BACKBONE == "mobilenetv2":
            self.prep = tf.keras.applications.mobilenet_v2.preprocess_input
        else:
            raise ValueError("BACKBONE must be 'efficientnetv2' or 'mobilenetv2'")

        print("Model input:", self.inp["shape"], "dtype:", self.inp["dtype"])
        print("Outputs:")
        for i, od in enumerate(self.out):
            print(f"  [{i}] name={od.get('name')} shape={od.get('shape')} dtype={od.get('dtype')}")
        self.map = None if len(self.out) == 1 else {
            i: next((h for h in TRAIN_HEADS if h in (od.get("name") or "").lower()), None)
            for i, od in enumerate(self.out)
        }
        print("Output mapping used:", self.map, "\n")

    def _x(self, p: Path):
        tf = self.tf
        b = tf.io.read_file(str(p))
        img = tf.image.decode_image(b, channels=3, expand_animations=False)  # uint8
        img = tf.image.resize(img, (self.h, self.w))
        img = tf.cast(img, tf.float32)

        if APPLY_EXIF_TRANSPOSE:
            import PIL.ImageOps as IOP
            arr = img.numpy().astype(np.uint8)
            img = tf.convert_to_tensor(np.asarray(IOP.exif_transpose(Image.fromarray(arr, "RGB")), np.float32))

        if self.inp["dtype"] == np.uint8:
            x = tf.cast(tf.clip_by_value(img, 0.0, 255.0), tf.uint8)[None, ...].numpy()
        else:
            x = self.prep(img)[None, ...].numpy()

        print(f"    input stats: min={float(x.min()):.3f} max={float(x.max()):.3f} mean={float(x.mean()):.3f}")
        return x

    def pred(self, p: Path):
        x = self._x(p)
        t0 = time.perf_counter()
        self.t.set_tensor(self.inp["index"], x)
        self.t.invoke()
        ms = (time.perf_counter() - t0) * 1000.0

        if len(self.out) == 1:
            y = np.asarray(self.t.get_tensor(self.out[0]["index"])).reshape(-1)
            return {TRAIN_HEADS[i]: float(y[i]) for i in range(min(4, y.shape[0]))}, ms

        scores = {}
        for i, od in enumerate(self.out):
            v = float(np.asarray(self.t.get_tensor(od["index"])).reshape(-1)[0])
            h = self.map.get(i) if self.map else None
            scores[h or (TRAIN_HEADS[i] if i < 4 else f"out{i}")] = v
        return scores, ms

# --- upload ---
def upload(layer, p: Path, lat, lon, field_scores: dict):
    geom = Point({"x": lon, "y": lat, "spatialReference": {"wkid": 4326}})
    attrs = {"image_file": p.name, "photo_date": int(os.path.getmtime(p) * 1000), **field_scores}
    r = layer.edit_features(adds=[Feature(geometry=geom, attributes=attrs)])
    return bool(r.get("addResults") and r["addResults"][0].get("success"))

# --- main ---
def main():
    gis = GIS(PORTAL, client_id=CLIENT_ID)
    print("Connected as:", gis.users.me.username)
    layer = get_layer(gis)
    m = M(MODEL_PATH)

    imgs = sorted([p for p in IMAGES_DIR.rglob("*") if p.suffix.lower() in EXTS])
    print(f"Found {len(imgs)} images in {IMAGES_DIR}\n")

    up = sk = er = 0
    for p in imgs:
        print("Processing:", p.name)
        lat, lon = gps(p)
        if lat is None or lon is None:
            print("    GPS: none (skip)\n")
            sk += 1
            continue
        print(f"    GPS: ({lat}, {lon})")

        try:
            head_scores, ms = m.pred(p)
            field_scores = {HEAD_TO_FIELD[h]: float(v) for h, v in head_scores.items() if h in HEAD_TO_FIELD}

            # fallback if name mapping incomplete
            if len(field_scores) < 4:
                for h in TRAIN_HEADS:
                    if h in head_scores and HEAD_TO_FIELD[h] not in field_scores:
                        field_scores[HEAD_TO_FIELD[h]] = float(head_scores[h])

            print("    scores(fields):", {k: round(v, 4) for k, v in field_scores.items()})
            print("    sorted(fields):", ", ".join(
                f"{k}={v:.4f}" for k, v in sorted(field_scores.items(), key=lambda kv: kv[1], reverse=True)
            ))
            print(f"    infer: {ms:.1f} ms")

            ok = upload(layer, p, lat, lon, field_scores)
            print("    upload:", "OK\n" if ok else "FAIL\n")
            up += int(ok)
        except Exception as e:
            er += 1
            print("    ERROR:", repr(e), "\n")

    print(f"Done. uploaded={up} skipped_no_gps={sk} errors={er}")

if __name__ == "__main__":
    main()
