import os, re, time
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image, ExifTags

from arcgis.gis import GIS
from arcgis.features import FeatureLayerCollection, Feature
from arcgis.geometry import Point

# ── SETTINGS ────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
MODEL_PATH = BASE / "model" / "model.tflite"
IMAGES_DIR  = BASE / "images"
EXTS = {".jpg",".jpeg",".png",".tif",".tiff",".heic"}

PORTAL="https://aoslcps.maps.arcgis.com"
CLIENT_ID="CpOBbpWXL80dumtM"
TITLE="TrailSense Conditions"

BACKBONE="efficientnetv2"      # "efficientnetv2" or "mobilenetv2" (must match training)
APPLY_EXIF_TRANSPOSE=False     # keep False if training used tf.image.decode_image (typical)

HEADS=["incision","muddiness","roots","braiding"]
MAP={"incision":"erosion","muddiness":"mud","roots":"roots","braiding":"braiding"}
FIELDS=["erosion","mud","roots","braiding"]

POLL=3
WORLD={"xmin":-180,"ymin":-90,"xmax":180,"ymax":90,"spatialReference":{"wkid":4326}}
GPS_TAG = next(k for k,v in ExifTags.TAGS.items() if v=="GPSInfo")


# ── GPS ─────────────────────────────────────────────────────────────────────
def _rat(x):
    try: return float(x)
    except Exception:
        try: return float(x[0])/float(x[1])
        except Exception: return None

def _dms(dms, ref):
    d,m,s=_rat(dms[0]),_rat(dms[1]),_rat(dms[2])
    if None in (d,m,s): return None
    v=d+m/60+s/3600
    return round(-v if ref in ("S","W") else v, 7)

def gps(p: Path):
    try:
        ex=Image.open(p)._getexif() or {}
        g=ex.get(GPS_TAG)
        if not g: return None,None
        return _dms(g[2],g[1]), _dms(g[4],g[3])  # lat, lon
    except Exception:
        return None,None


# ── MODEL (correct TF preprocessing + correct head mapping) ──────────────────
class Model:
    def __init__(self, tflite_path: Path):
        import tensorflow as tf
        self.tf=tf
        self.t=tf.lite.Interpreter(model_path=str(tflite_path))
        self.t.allocate_tensors()
        self.inp=self.t.get_input_details()[0]
        self.out=self.t.get_output_details()
        _,h,w,_=self.inp["shape"]
        self.h,self.w=int(h),int(w)

        if BACKBONE=="efficientnetv2":
            self.prep=tf.keras.applications.efficientnet_v2.preprocess_input
        elif BACKBONE=="mobilenetv2":
            self.prep=tf.keras.applications.mobilenet_v2.preprocess_input
        else:
            raise ValueError("BACKBONE must be 'efficientnetv2' or 'mobilenetv2'")

        self.name_map = None if len(self.out)==1 else {
            i: next((hh for hh in HEADS if hh in (od.get("name") or "").lower()), None)
            for i,od in enumerate(self.out)
        }

        print("Model input:", self.inp["shape"], "dtype:", self.inp["dtype"])
        print("Output mapping:", self.name_map, "\n")

    def _x(self, p: Path):
        tf=self.tf
        b=tf.io.read_file(str(p))
        img=tf.image.decode_image(b, channels=3, expand_animations=False)
        img=tf.image.resize(img, (self.h,self.w))
        img=tf.cast(img, tf.float32)

        if APPLY_EXIF_TRANSPOSE:
            import PIL.ImageOps as IOP
            arr=img.numpy().astype(np.uint8)
            img=tf.convert_to_tensor(np.asarray(IOP.exif_transpose(Image.fromarray(arr,"RGB")), np.float32))

        if self.inp["dtype"]==np.uint8:
            x=tf.cast(tf.clip_by_value(img,0,255), tf.uint8)[None,...].numpy()
        else:
            x=self.prep(img)[None,...].numpy()

        print(f"    input stats: min={float(x.min()):.3f} max={float(x.max()):.3f} mean={float(x.mean()):.3f}")
        return x

    def predict(self, p: Path):
        x=self._x(p)
        t0=time.perf_counter()
        self.t.set_tensor(self.inp["index"], x)
        self.t.invoke()
        ms=(time.perf_counter()-t0)*1000

        if len(self.out)==1:
            y=np.asarray(self.t.get_tensor(self.out[0]["index"])).reshape(-1)
            return {HEADS[i]: float(y[i]) for i in range(min(4,y.shape[0]))}, ms

        s={}
        for i,od in enumerate(self.out):
            v=float(np.asarray(self.t.get_tensor(od["index"])).reshape(-1)[0])
            h=self.name_map.get(i) if self.name_map else None
            s[h or (HEADS[i] if i<4 else f"out{i}")]=v
        return s, ms


# ── ARCGIS: create/get + patch SR/extent + ensure fields ─────────────────────
def _ensure_fields(layer):
    want = ["image_file","photo_date","latitude","longitude",*FIELDS]
    have = {f.get("name","").lower() for f in layer.properties.fields}
    add=[]
    for n in want:
        if n.lower() in have: continue
        if n in ("image_file",):
            add.append({"name":n,"alias":n,"type":"esriFieldTypeString","length":255})
        elif n in ("photo_date",):
            add.append({"name":n,"alias":n,"type":"esriFieldTypeDate"})
        else:
            add.append({"name":n,"alias":n,"type":"esriFieldTypeDouble"})
    if add:
        layer.manager.add_to_definition({"fields": add})

def _patch(item, layer):
    try:
        FeatureLayerCollection.fromitem(item).manager.update_definition({
            "spatialReference":{"wkid":4326},
            "fullExtent":WORLD,
            "initialExtent":WORLD,
        })
    except Exception:
        pass
    try:
        layer.manager.update_definition({"spatialReference":{"wkid":4326}, "extent":WORLD})
    except Exception:
        pass

def _has_geom(layer):
    try:
        fs = layer.query(where="1=1", out_fields=layer.properties.objectIdField,
                         return_geometry=True, result_record_count=5).features
        return any((f.geometry or {}).get("x") is not None and (f.geometry or {}).get("y") is not None for f in fs)
    except Exception:
        return False

def _create(gis: GIS):
    svc = re.sub(r"[^A-Za-z0-9_]+","_",TITLE).strip("_")[:40] + "_" + datetime.utcnow().strftime("%Y%m%d%H%M%S")
    item = gis.content.create_service(
        name=svc, service_type="featureService",
        create_params={"name":svc,"capabilities":"Query,Create,Update,Delete,Editing","spatialReference":{"wkid":4326}},
    )
    item.update(item_properties={"title": TITLE})
    FeatureLayerCollection.fromitem(item).manager.add_to_definition({
        "fullExtent": WORLD, "initialExtent": WORLD,
        "layers":[{
            "id":0, "name":"observations", "type":"Feature Layer",
            "geometryType":"esriGeometryPoint",
            "objectIdField":"OBJECTID",
            "spatialReference":{"wkid":4326},
            "extent": WORLD,
            "fields":[
                {"name":"OBJECTID","alias":"OBJECTID","type":"esriFieldTypeOID"},
                {"name":"image_file","alias":"image_file","type":"esriFieldTypeString","length":255},
                {"name":"photo_date","alias":"photo_date","type":"esriFieldTypeDate"},
                {"name":"latitude","alias":"latitude","type":"esriFieldTypeDouble"},
                {"name":"longitude","alias":"longitude","type":"esriFieldTypeDouble"},
                *[{"name":f,"alias":f,"type":"esriFieldTypeDouble"} for f in FIELDS],
            ],
        }]
    })
    for _ in range(15):
        it = gis.content.get(item.id)
        try:
            layer = it.layers[0]
            _ensure_fields(layer); _patch(it, layer)
            return it, layer
        except Exception:
            time.sleep(1)
    raise RuntimeError("Created service but layer not available yet.")

def get_or_create(gis: GIS):
    me = gis.users.me.username
    items = gis.content.search(f'title:"{TITLE}" AND owner:{me}', "Feature Layer", max_items=10)
    if not items:
        print("Layer not found — creating…")
        return _create(gis)

    item = items[0]
    layer = item.layers[0]
    _ensure_fields(layer); _patch(item, layer)
    print("Layer found:", item.title)
    return item, layer


# ── REPAIR existing rows: ensure geometry is valid (wkid 4326) ───────────────
def repair(layer, images_by_name):
    oid = layer.properties.objectIdField
    fs = layer.query(where="1=1", out_fields=f"{oid},image_file,latitude,longitude",
                     return_geometry=True, result_record_count=5000).features
    upd=[]
    for f in fs:
        g=f.geometry or {}
        lat=f.attributes.get("latitude")
        lon=f.attributes.get("longitude")
        name=f.attributes.get("image_file")

        if (lat is None or lon is None) and name in images_by_name:
            lat, lon = gps(images_by_name[name])
        if lat is None or lon is None:
            continue

        x,y = g.get("x"), g.get("y")
        bad = (x is None or y is None)
        if not bad and (g.get("spatialReference") or {}).get("wkid") != 4326:
            bad = True

        if bad:
            f.geometry={"x":float(lon),"y":float(lat),"spatialReference":{"wkid":4326}}
            f.attributes={oid:f.attributes[oid],"latitude":float(lat),"longitude":float(lon)}
            upd.append(f)

    if upd:
        print(f"Repairing {len(upd)} geometries…")
        for i in range(0, len(upd), 200):
            layer.edit_features(updates=upd[i:i+200])


# ── ADD feature (always explicit 4326 geom + lat/lon attrs) ──────────────────
def add(layer, p: Path, lat, lon, vals: dict):
    geom={"x":float(lon),"y":float(lat),"spatialReference":{"wkid":4326}}
    attrs={
        "image_file": p.name,
        "photo_date": int(os.path.getmtime(p)*1000),
        "latitude": float(lat),
        "longitude": float(lon),
        **vals
    }
    r = layer.edit_features(adds=[Feature(geometry=geom, attributes=attrs)])
    return bool(r.get("addResults") and r["addResults"][0].get("success"))


# ── WATCH folder ────────────────────────────────────────────────────────────
def watch():
    seen=set()
    while True:
        for p in sorted([p for p in IMAGES_DIR.rglob("*") if p.suffix.lower() in EXTS]):
            k=(str(p.resolve()), int(p.stat().st_mtime), p.stat().st_size)
            if k in seen: continue
            seen.add(k)
            yield p
        time.sleep(POLL)


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    gis = GIS(PORTAL, client_id=CLIENT_ID)
    print("Connected as:", gis.users.me.username)

    item, layer = get_or_create(gis)

    imgs = {p.name: p for p in IMAGES_DIR.rglob("*") if p.suffix.lower() in EXTS}
    repair(layer, imgs)

    # if the existing service is fundamentally broken (no drawable geometry), archive + recreate clean
    if not _has_geom(layer):
        try:
            item.update(item_properties={"title": f"{TITLE} (archived {datetime.now().strftime('%Y-%m-%d %H%M')})"})
        except Exception:
            pass
        item, layer = _create(gis)

    model = Model(MODEL_PATH)
    print(f"Watching: {IMAGES_DIR} (every {POLL}s)\n")

    for p in watch():
        print("Processing:", p.name)
        lat, lon = gps(p)
        if lat is None or lon is None:
            print("    GPS: none (skip)\n")
            continue
        print(f"    GPS: ({lat}, {lon})")

        hs, ms = model.predict(p)
        vals = {MAP[h]: float(v) for h,v in hs.items() if h in MAP}
        if len(vals) < 4:
            for h in HEADS:
                if h in hs and MAP[h] not in vals:
                    vals[MAP[h]] = float(hs[h])

        print("    scores:", {k: round(v,4) for k,v in vals.items()})
        print("    sorted:", ", ".join(f"{k}={v:.4f}" for k,v in sorted(vals.items(), key=lambda kv: kv[1], reverse=True)))
        print(f"    infer: {ms:.1f} ms")

        ok = add(layer, p, lat, lon, vals)
        print("    upload:", "OK\n" if ok else "FAIL\n")

if __name__ == "__main__":
    main()
