"""
Improved Document & Photo Text Extractor
- Adds OpenCV-based preprocessing for photographs: deskew, denoise, adaptive threshold
- Adds parallel OCR for images/pages
- New CLI flags: --photo-max-width, --deskew, --denoise, --adaptive-thresh, --threads, --dpi

Usage:
python extract_text_improved.py --input path --out out_dir --ocr --deskew --denoise --threads 4
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, List

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None

from concurrent.futures import ThreadPoolExecutor, as_completed
import math


def _pil_from_cv2(img_cv2):
    # cv2 image BGR or single channel to PIL
    if img_cv2 is None:
        return None
    if len(img_cv2.shape) == 2:
        mode = 'L'
    else:
        mode = 'RGB'
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_cv2)


def _rotate_image(img_cv2, angle_deg):
    (h, w) = img_cv2.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(img_cv2, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def detect_orientation_by_osd(pil_img) -> int:
    """Return rotation angle (clockwise) recommended by Tesseract OSD, 0/90/180/270"""
    if pytesseract is None:
        return 0
    try:
        osd = pytesseract.image_to_osd(pil_img)
        for line in osd.splitlines():
            if line.strip().startswith('Rotate:'):
                angle = int(line.split(':')[1].strip())
                return angle
    except Exception as e:
        logging.debug(f"OSD failed: {e}")
    return 0


def deskew_cv2(gray):
    """Estimate rotation using moments of edges and rotate to deskew"""
    if cv2 is None or np is None:
        return gray
    # threshold
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 10:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:
        return gray
    logging.debug(f"Deskewing by {angle:.2f} degrees")
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def preprocess_photo(path_or_pil, max_width: Optional[int] = 1800, denoise: bool = True, deskew: bool = True, adaptive_thresh: bool = True, detect_orientation: bool = True):
    """Return a PIL image ready for pytesseract from a photo.
    Accepts either path (str/Path) or a PIL Image.
    """
    # load
    if isinstance(path_or_pil, (str, Path)):
        if cv2 is None:
            raise RuntimeError("opencv-python is required for photo preprocessing")
        img_cv2 = cv2.imread(str(path_or_pil))
        if img_cv2 is None:
            raise RuntimeError(f"Unable to open image: {path_or_pil}")
    else:
        pil_img = path_or_pil
        if Image is None:
            raise RuntimeError("Pillow is required")
        img_cv2 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # resize
    h, w = img_cv2.shape[:2]
    if max_width and w > max_width:
        scale = max_width / float(w)
        img_cv2 = cv2.resize(img_cv2, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

    if denoise:
        # fastNlMeansDenoising for grayscale
        gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    if deskew:
        try:
            gray = deskew_cv2(gray)
        except Exception as e:
            logging.debug(f"Deskew failed: {e}")

    if adaptive_thresh:
        try:
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        except Exception as e:
            logging.debug(f"Adaptive thresholding failed: {e}")
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    pil_out = _pil_from_cv2(gray)

    # orientation detection (OSD): rotate to upright
    if detect_orientation and pytesseract is not None:
        try:
            angle = detect_orientation_by_osd(pil_out)
            if angle:
                logging.debug(f"OSD suggests rotation {angle}")
                pil_out = pil_out.rotate(angle, expand=True)
        except Exception as e:
            logging.debug(f"OSD detect error: {e}")

    return pil_out


def ocr_pil_image(pil_img, lang='eng'):
    if pytesseract is None:
        raise RuntimeError("pytesseract is required for OCR")
    return pytesseract.image_to_string(pil_img, lang=lang)


def extract_text_from_image_advanced(path: str, *, max_width: int = 1800, denoise: bool = True, deskew: bool = True, adaptive_thresh: bool = True, lang: str = 'eng') -> str:
    pil = preprocess_photo(path, max_width=max_width, denoise=denoise, deskew=deskew, adaptive_thresh=adaptive_thresh)
    return ocr_pil_image(pil, lang=lang)


def extract_text_from_pdf_advanced(path: str, use_ocr: bool = False, dpi: int = 200, threads: int = 1, **photo_opts) -> str:
    # Use pdf2image to convert to images and run preprocess + OCR in parallel
    text_parts = []
    if pdfplumber := None:  # keep previous pdf text extraction optional; left out here for clarity
        pass
    if convert_from_path is None:
        raise RuntimeError("pdf2image is required for PDF page conversion to images")

    images = convert_from_path(path, dpi=dpi)

    # If threads==1, run sequentially
    if threads <= 1:
        for img in images:
            try:
                pil = preprocess_photo(img, **photo_opts)
                text_parts.append(ocr_pil_image(pil, lang=photo_opts.get('lang', 'eng')))
            except Exception as e:
                logging.debug(f"page OCR error: {e}")
    else:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            futures = [ex.submit(lambda im: ocr_pil_image(preprocess_photo(im, **photo_opts), lang=photo_opts.get('lang', 'eng')), img) for img in images]
            for fut in as_completed(futures):
                try:
                    text_parts.append(fut.result())
                except Exception as e:
                    logging.debug(f"page OCR parallel error: {e}")

    return "\n".join(t for t in text_parts if t)


def main():
    p = argparse.ArgumentParser(description='Improved document / photo text extractor')
    p.add_argument('-i', '--input', required=True)
    p.add_argument('-o', '--out', default='./extracted_text')
    p.add_argument('--ocr', action='store_true', help='Force OCR for PDFs')
    p.add_argument('--lang', default='eng')
    p.add_argument('--dpi', type=int, default=200, help='DPI for PDF->image (lower is faster)')
    p.add_argument('--threads', type=int, default=1, help='Parallel OCR threads for images/pages')

    # photo options
    p.add_argument('--photo-max-width', type=int, default=1800, help='Resize larger photos to this width before OCR')
    p.add_argument('--deskew', action='store_true', help='Attempt to deskew photos')
    p.add_argument('--denoise', action='store_true', help='Apply denoising filter to photos')
    p.add_argument('--adaptive-thresh', action='store_true', help='Use adaptive thresholding to binarize photos (helps OCR)')
    p.add_argument('-v', '--verbose', action='store_true')

    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='%(levelname)s: %(message)s')

    inp = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if inp.is_file():
        ext = inp.suffix.lower()
        if ext == '.pdf':
            txt = extract_text_from_pdf_advanced(str(inp), use_ocr=args.ocr, dpi=args.dpi, threads=args.threads, max_width=args.photo_max_width, denoise=args.denoise, deskew=args.deskew, adaptive_thresh=args.adaptive_thresh, lang=args.lang)
            (out_dir / (inp.stem + '.txt')).write_text(txt or '', encoding='utf-8')
            logging.info(f"Saved: {(out_dir / (inp.stem + '.txt'))}")
        elif ext in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}:
            txt = extract_text_from_image_advanced(str(inp), max_width=args.photo_max_width, denoise=args.denoise, deskew=args.deskew, adaptive_thresh=args.adaptive_thresh, lang=args.lang)
            (out_dir / (inp.stem + '.txt')).write_text(txt or '', encoding='utf-8')
            logging.info(f"Saved: {(out_dir / (inp.stem + '.txt'))}")
        else:
            logging.error('Unsupported file type for improved extractor')
    else:
        logging.error('Input must be a file (no directory recursion in improved script)')


if __name__ == '__main__':
    main()
