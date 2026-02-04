
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from arcgis.gis import GIS
from arcgis.features import FeatureLayer, Feature
from arcgis.geometry import Point

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent / "model" / "model.tflite"
IMAGES_DIR = Path(__file__).parent / "images"

LABELS = ["erosion", "mud", "roots", "braiding"]
CONFIDENCE_THRESHOLD = 0.5
INPUT_SIZE = (224, 224)

# ArcGIS
PORTAL = "https://aoslcps.maps.arcgis.com"
CLIENT_ID = "CpOBbpWXL80dumtM"
FEATURE_LAYER_URL = "https://services4.arcgis.com/9moZ1UwKSaAK9hCA/arcgis/rest/services/test/FeatureServer/0"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".heic"}


# ── GPS Extraction ───────────────────────────────────────────────────────────

def _dms_to_decimal(dms, ref):
    degrees = float(dms[0])
    minutes = float(dms[1])
    seconds = float(dms[2])
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in ("S", "W"):
        decimal = -decimal
    return round(decimal, 7)


def extract_gps(image_path):
    """Pull lat/lon from EXIF. Returns (lat, lon) or (None, None)."""
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data is None:
            return None, None

        gps_info = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "GPSInfo":
                for gps_tag_id, gps_value in value.items():
                    gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info[gps_tag] = gps_value

        if not gps_info:
            return None, None

        lat = _dms_to_decimal(gps_info["GPSLatitude"], gps_info["GPSLatitudeRef"])
        lon = _dms_to_decimal(gps_info["GPSLongitude"], gps_info["GPSLongitudeRef"])
        return lat, lon
    except Exception:
        return None, None


def extract_timestamp(image_path):
    """Pull date taken from EXIF, fallback to file mod time."""
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "DateTimeOriginal":
                    return datetime.strptime(value, "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        pass
    mtime = os.path.getmtime(image_path)
    return datetime.fromtimestamp(mtime, tz=timezone.utc)


# ── Model ────────────────────────────────────────────────────────────────────

class TrailClassifier:
    def __init__(self, model_path=MODEL_PATH):
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        shape = self.input_details[0]["shape"]
        self.input_size = (shape[1], shape[2])
        print(f"Model loaded: {self.input_size}")

    def classify(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img = img.resize(self.input_size, Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

        self.interpreter.set_tensor(self.input_details[0]["index"], arr)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        # sigmoid if needed
        if output.max() > 1.0 or output.min() < 0.0:
            output = 1.0 / (1.0 + np.exp(-output))
        print(f"Output shape: {output.shape}, Output: {output}")

        scores = {label: round(float(output[i]), 4) for i, label in enumerate(LABELS)}
        flagged = [l for l, s in scores.items() if s >= CONFIDENCE_THRESHOLD]
        return scores, flagged


# ── ArcGIS ───────────────────────────────────────────────────────────────────

def connect_arcgis():
    """Authenticate via OAuth and return (gis, layer)."""
    print("Connecting to ArcGIS...")
    print("A browser window will open for sign-in.")
    gis = GIS(PORTAL, client_id=CLIENT_ID)
    print(f"Connected as: {gis.users.me.username}")
    layer = FeatureLayer(FEATURE_LAYER_URL)
    return gis, layer


def upload_result(layer, result):
    """Push a single result to ArcGIS."""
    if result["lat"] is None:
        print(f"  Skipping {result['file']} — no GPS")
        return

    point = Point({
        "x": result["lon"],
        "y": result["lat"],
        "spatialReference": {"wkid": 4326},
    })

    attributes = {
        "image_file": result["file"],
        "photo_date": int(result["timestamp"].timestamp() * 1000),
        "latitude": result["lat"],
        "longitude": result["lon"],
        "erosion": result["scores"]["erosion"],
        "mud": result["scores"]["mud"],
        "roots": result["scores"]["roots"],
        "braiding": result["scores"]["braiding"],
        "flagged": ", ".join(result["flagged"]) if result["flagged"] else "none",
    }

    feature = Feature(geometry=point, attributes=attributes)
    res = layer.edit_features(adds=[feature])
    
    if res["addResults"][0]["success"]:
        print(f"  ✓ Uploaded {result['file']}")
    else:
        print(f"  ✗ Failed {result['file']}: {res['addResults'][0]}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n=== TrailSense ===\n")

    # Connect to ArcGIS
    gis, layer = connect_arcgis()

    # Load model
    print("\nLoading model...")
    classifier = TrailClassifier()

    # Find images
    images = [p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    print(f"\nFound {len(images)} images in {IMAGES_DIR}\n")

    # Process each image
    for img_path in sorted(images):
        print(f"Processing: {img_path.name}")
        
        lat, lon = extract_gps(img_path)
        timestamp = extract_timestamp(img_path)
        scores, flagged = classifier.classify(img_path)

        result = {
            "file": img_path.name,
            "lat": lat,
            "lon": lon,
            "timestamp": timestamp,
            "scores": scores,
            "flagged": flagged,
        }

        # Print result
        gps_str = f"({lat}, {lon})" if lat else "no GPS"
        flag_str = ", ".join(flagged) if flagged else "none"
        print(f"  GPS: {gps_str}")
        print(f"  Scores: {scores}")
        print(f"  Flagged: {flag_str}")

        # Upload
        upload_result(layer, result)
        print()

    print("Done.")


if __name__ == "__main__":
    main()