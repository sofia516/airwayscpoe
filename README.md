# AirwayScope

AirwayScope is an academic research demo that estimates a small set of facial dimensions from a photograph. It is experimental, not clinically validated, and must not be used to guide diagnosis or treatment.

## Run locally

Requires Python 3.10 or newer.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser. Choose **Upload photo** or **Use camera**, then select a supported image (JPEG, PNG, or WebP). The short side of the image must be at least 480 pixels.

MediaPipe Face Mesh uses its refined iris landmarks to support approximate iris calibration. The dependency is kept below version 0.10.22 because later releases removed the legacy `mp.solutions` API used here.

## Photo capture protocol

For more consistent estimates:

1. Place the camera on a stable surface or tripod and keep the camera distance consistent between photos.
2. Use even, front-facing light and a plain background. Avoid strong shadows, filters, and digital zoom.
3. Keep the full face in frame and face the camera with the head level and expression neutral.
4. Open the mouth as wide as comfortable; do not force it.
5. For preferred scale calibration, include an object of known length (such as a small ruler) close to the face and in the same plane. Avoid placing it closer to the camera than the face.
6. Use an image whose shorter dimension is at least 480 pixels.

With **Known reference object** calibration, enter the object's length in millimetres and the pixel coordinates of its two endpoints. **Iris estimate** is faster but approximate (about 10%). Face pose, perspective, lighting, occlusion, and expression can affect landmarks and measurements.

## Measurements

- **Mouth opening:** distance between inner upper and lower lip landmarks (MediaPipe points 13 and 14), in millimetres.
- **Mouth width:** distance between mouth corners (points 61 and 291), in millimetres.
- **Jaw width:** distance between jaw landmarks (points 234 and 454), in millimetres.
- **Face height proxy:** distance from forehead proxy (point 10) to chin (point 152), in millimetres. This is not anatomical total face height.
- **Opening-to-height ratio:** mouth opening divided by the face height proxy; unitless.

All linear measurements are Euclidean pixel distances multiplied by the selected calibration scale. The largest detected face is measured when an image contains multiple faces. Quality warnings are heuristics, not validated acceptance criteria.

## Example result

The values below are **illustrative sample data only**. They are not measurements from a real person, an uploaded photo, or a validated study.

| Measurement | Example value |
| --- | ---: |
| Mouth opening | 8.4 mm |
| Mouth width | 48.3 mm |
| Jaw width | 112.7 mm |
| Face height proxy | 172.4 mm |
| Opening-to-height ratio | 0.049 |

In the app, saved results appear in **Saved records** and can be downloaded as a CSV file. The sample values in this README are not added to the app database or export.

## Records and CSV export

Choose **Save this measurement** to save a generated anonymous case code, measurements, calibration label, quality warnings, and timestamp in SQLite. The **Saved records** tab shows stored cases and provides **Download CSV** for export. The image itself is processed in memory and is not saved.

Records are stored in `airwayscope_records.sqlite3` beside the app. Set `AIRWAYSCOPE_DB` to use another database file. The database is local to the machine unless you move or share it.

## Tests

```powershell
python -m pytest -q
```

## Data and ethics

Obtain informed consent before capturing or processing an image. Do not enter names or other identifying details. Follow your institution's ethics review and data-governance requirements for human-subject research. Delete local database files when they are no longer needed. This demo is not a diagnostic tool and must not be used for clinical decisions.
