
import base64, hmac, io, json, os, time
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics import YOLO

Image.MAX_IMAGE_PIXELS = 40_000_000
MAX_BYTES = 5 * 1024 * 1024
FORMATS = {"JPEG", "PNG", "WEBP"}
REPO, API_TOKEN, READ = os.environ["MODEL_REPO"], os.environ["API_TOKEN"], os.environ.get("HF_READ_TOKEN")
CFG = json.load(open(hf_hub_download(REPO, "config.json", token=READ)))
MODEL = YOLO(hf_hub_download(REPO, "model.onnx", token=READ), task="segment")
app = FastAPI(title="Ayomi Foto Fokus Anak", version=CFG["model_version"])

def fail(status, code, message):
    raise HTTPException(status_code=status, detail={"code": code, "error": message})

def authorize(header):
    if not hmac.compare_digest(header or "", f"Bearer {API_TOKEN}"):
        fail(401, "unauthorized", "Token API tidak valid.")

async def read_image(file):
    data = await file.read(MAX_BYTES + 1)
    if not data: fail(400, "empty_file", "Berkas kosong.")
    if len(data) > MAX_BYTES: fail(413, "file_too_large", "Foto maksimal 5 MB.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            img = ImageOps.exif_transpose(im).convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
        fail(400, "invalid_image", "Berkas bukan gambar yang dapat dibaca.")
    if fmt not in FORMATS: fail(415, "unsupported_format", "Format harus JPG, PNG, atau WebP.")
    img.thumbnail((CFG["max_side"], CFG["max_side"]))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def poly_mask(poly, W, H):
    m = np.zeros((H, W), np.uint8)
    if len(poly) >= 3: cv2.fillPoly(m, [np.round(np.asarray(poly)).astype(np.int32)], 1)
    return m.astype(bool)

def mask_iou(a, b):
    union = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / union if union else 0.0

def segment(bgr):
    H, W = bgr.shape[:2]
    r = MODEL.predict(bgr, conf=CFG["conf_threshold"], iou=0.6, imgsz=CFG["imgsz"], max_det=50, verbose=False)[0]
    people = []
    if r.masks is not None:
        for poly, conf, cls in zip(r.masks.xy, r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            if len(poly) < 3: continue
            p = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
            p = cv2.approxPolyDP(p, 0.003 * cv2.arcLength(p, True), True).reshape(-1, 2)
            if len(p) < 3: continue
            p[:, 0] = np.clip(p[:, 0], 0, W - 1); p[:, 1] = np.clip(p[:, 1], 0, H - 1)
            people.append({"class": CFG["classes"][int(cls)], "confidence": round(float(conf), 4),
                           "polygon": np.round(p).astype(int).tolist()})
    people.sort(key=lambda q: -q["confidence"])
    for i, q in enumerate(people): q["id"] = i
    return people

def render_focus(bgr, target_mask, other_masks, extra_boxes, cfg):
    H, W = bgr.shape[:2]
    k = max(31, int(min(H, W) * cfg["blur_frac"]) | 1)
    background = cv2.GaussianBlur(bgr, (k, k), 0)
    kernel = np.ones((max(3, int(max(H, W) * cfg["dilate_frac"])),) * 2, np.uint8)
    keep = cv2.dilate(target_mask.astype(np.uint8), kernel)
    others = np.zeros((H, W), np.uint8)
    for m in other_masks: others |= m.astype(np.uint8)
    others = cv2.dilate(others, kernel).astype(bool)
    if others.any():
        b = cfg["pixel_block"]
        small = cv2.resize(background, (max(1, W // b), max(1, H // b)), interpolation=cv2.INTER_AREA)
        background[others] = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)[others]
    for x, y, w, h in extra_boxes:
        keep[max(0, int(y)):int(y + h), max(0, int(x)):int(x + w)] = 0
    f = max(3, int(max(H, W) * cfg["feather_frac"]) | 1)
    alpha = cv2.GaussianBlur(keep.astype(np.float32), (f, f), 0)[..., None]
    out = (alpha * bgr + (1 - alpha) * background).astype(np.uint8)
    keep_b = keep.astype(bool)
    overlap = any(m.sum() and np.logical_and(m, keep_b).sum() / m.sum() > CFG["leak_ratio"] for m in other_masks)
    return out, (["overlap_other_person"] if overlap else [])

def quality(bgr, people):
    H, W = bgr.shape[:2]
    flags = []
    if not people: flags.append("no_person_found")
    if any(q["confidence"] < CFG["auto_conf"] for q in people): flags.append("low_confidence")
    if sum(q["class"] == "anak" for q in people) > 1: flags.append("multiple_children")
    if min(W, H) < 320: flags.append("small_image")
    if cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < 60: flags.append("blurry")
    return flags

@app.get("/health")
def health():
    return {"status": "ok", "model_version": CFG["model_version"]}

@app.post("/segment")
async def segment_photo(file: UploadFile = File(...), authorization: str | None = Header(default=None)):
    authorize(authorization)
    bgr = await read_image(file); started = time.perf_counter()
    H, W = bgr.shape[:2]
    people = segment(bgr)
    kids = [q for q in people if q["class"] == "anak"]
    suggested = kids[0]["id"] if len(kids) == 1 and kids[0]["confidence"] >= CFG["auto_conf"] else None
    return {"image_size": {"width": W, "height": H}, "people": people, "suggested_target_id": suggested,
            "quality_flags": quality(bgr, people), "requires_human_review": True,
            "model_version": CFG["model_version"], "latency_ms": round((time.perf_counter() - started) * 1000)}

@app.post("/render")
async def render_photo(file: UploadFile = File(...), target_polygon: str = Form(...),
                       extra_blur_boxes: str = Form("[]"), authorization: str | None = Header(default=None)):
    authorize(authorization)
    bgr = await read_image(file); started = time.perf_counter()
    H, W = bgr.shape[:2]
    try:
        target = np.asarray(json.loads(target_polygon), dtype=float)
        boxes = json.loads(extra_blur_boxes)
        valid = (target.ndim == 2 and target.shape[1] == 2 and 3 <= len(target) <= 2000
                 and target[:, 0].min() >= 0 and target[:, 0].max() <= W
                 and target[:, 1].min() >= 0 and target[:, 1].max() <= H
                 and isinstance(boxes, list) and len(boxes) <= 20
                 and all(isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b) for b in boxes))
    except Exception:
        valid = False
    if not valid:
        fail(400, "invalid_target", "target_polygon atau extra_blur_boxes tidak valid untuk ukuran foto ini.")
    target_mask = poly_mask(target, W, H)
    others = [m for m in (poly_mask(q["polygon"], W, H) for q in segment(bgr)) if mask_iou(m, target_mask) < 0.5]
    image, flags = render_focus(bgr, target_mask, others, boxes, CFG["render"])
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])   # tanpa EXIF
    return {"image_size": {"width": W, "height": H}, "other_people_blurred": len(others),
            "image": {"mime_type": "image/jpeg", "base64": base64.b64encode(jpeg.tobytes()).decode()},
            "quality_flags": ["manual_review_required", *flags], "requires_human_review": True,
            "model_version": CFG["model_version"], "latency_ms": round((time.perf_counter() - started) * 1000)}
