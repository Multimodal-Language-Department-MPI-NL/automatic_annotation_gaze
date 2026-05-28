#!/usr/bin/env python3
import argparse, csv, json, math, os, sys
from typing import Iterable, List, Optional, Tuple
import logging
import tempfile
logger = logging.getLogger(__name__)

Sample = Tuple[Optional[float], float, float, float]  # (t_seconds or None, x, y, conf)

# --- helpers -----------------------------------------------------------------

def _as_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default

def read_pupil_json(path: str) -> List[Sample]:
    """
    Supports newline-delimited JSON or a JSON array.
    Accepts common Pupil/Neon keys:
      - time: 'timestamp' (sec) or 'ts' (microseconds)
      - pos:  'gp' (list[x,y]) or 'gaze2d' or 'norm_pos'
      - conf: 'confidence' or 'l' or 'c'
      - state: 's' (0 means OK in many exports)
    """
    def parse_obj(o) -> Optional[Sample]:
        # time
        t = None
        if 'timestamp' in o:
            t = _as_float(o['timestamp'])
        elif 'ts' in o:
            us = _as_float(o['ts'])
            if us is not None:
                t = us / 1_000_000.0

        # position
        x = y = None
        if 'gp' in o and isinstance(o['gp'], (list, tuple)) and len(o['gp']) >= 2:
            x, y = _as_float(o['gp'][0]), _as_float(o['gp'][1])
        elif 'gaze2d' in o.get('data', {}) and isinstance(o['data']['gaze2d'], (list, tuple)):
            x, y = _as_float(o['data']['gaze2d'][0]), _as_float(o['data']['gaze2d'][1])
        elif 'norm_pos' in o and isinstance(o['norm_pos'], (list, tuple)) and len(o['norm_pos']) >= 2:
            x, y = _as_float(o['norm_pos'][0]), _as_float(o['norm_pos'][1])

        # confidence
        conf = None
        for k in ('confidence','l','c'):
            if k in o:
                conf = _as_float(o[k])
                break
        if conf is None and 'data' in o and 'confidence' in o['data']:
            conf = _as_float(o['data']['confidence'])

        # validity/state
        if 's' in o:
            # many Pupil exports use s==0 as "good". Drop others.
            if _as_float(o['s'], -1) != 0:
                return None

        if x is None or y is None:
            return None
        if conf is None:
            conf = 1.0
        return (t, x, y, conf)

    samples: List[Sample] = []

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            data = json.load(f)
            for o in data:
                if isinstance(o, dict):
                    s = parse_obj(o)
                    if s:
                        samples.append(s)
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                if isinstance(o, dict):
                    s = parse_obj(o)
                    if s:
                        samples.append(s)

    return samples

def read_tobii_csv(path: str) -> List[Sample]:
    """
    Tries to be robust to common Tobii Pro Glasses 3 columns.
    Looks for likely timestamp/x/y/confidence/validity headers.
    """
    def pick(colnames, *candidates):
        cands = [c.lower() for c in candidates]
        for name in colnames:
            n = name.lower()
            for c in cands:
                if c in n:
                    return name
        return None

    samples: List[Sample] = []
    with open(path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        # timestamp
        t_col = pick(cols, 'timestamp', 'time', 'system time', 'scene time', 'recording time')
        # gaze position (normalized)
        x_col = pick(cols, 'gaze position x', 'gaze x', 'gp x', 'x [normalized]', 'x_norm', 'x')
        y_col = pick(cols, 'gaze position y', 'gaze y', 'gp y', 'y [normalized]', 'y_norm', 'y')
        # confidence / validity
        conf_col = pick(cols, 'confidence')
        valid_col = pick(cols, 'validity', 'valid', 'gaze point valid', 'gaze validity')

        for row in reader:
            t = _as_float(row.get(t_col), None) if t_col else None
            # Some Tobii exports use milliseconds since start; normalize to seconds if it looks large
            if t is not None and t > 10_000:  # likely ms
                t = t / 1000.0
            x = _as_float(row.get(x_col), None) if x_col else None
            y = _as_float(row.get(y_col), None) if y_col else None
            if x is None or y is None:
                continue

            conf = None
            if conf_col:
                conf = _as_float(row.get(conf_col), None)
            if conf is None and valid_col:
                v = row.get(valid_col)
                # map common validity encodings to [0..1]
                if v is not None:
                    try:
                        vi = float(v)
                        # many Tobii fields are 0/1/2; treat 0=good, else 0
                        conf = 1.0 if vi == 0 or vi == 1 else 0.0
                    except Exception:
                        conf = 1.0 if str(v).strip().lower() in ('true','valid','yes') else 0.0
            if conf is None:
                conf = 1.0

            samples.append((t, x, y, conf))

    return samples

def align_to_frames(samples: List[Sample], fps: Optional[float]) -> List[Tuple[int,float,float,float]]:
    """
    If fps is provided and timestamps exist, average samples into frame buckets.
    Otherwise, enumerate samples in order.
    """
    if not samples:
        return []

    if fps and any(s[0] is not None for s in samples):
        # keep only samples with timestamps
        ts_samples = [(t,x,y,c) for (t,x,y,c) in samples if t is not None]
        if not ts_samples:
            # fall back to enumeration
            return [(i, x, y, c) for i, (_,x,y,c) in enumerate(samples)]

        t0 = ts_samples[0][0]
        buckets = {}
        for (t,x,y,c) in ts_samples:
            fi = int(round((t - t0) * fps))
            if fi not in buckets:
                buckets[fi] = {'x':[], 'y':[], 'c':[]}
            buckets[fi]['x'].append(x)
            buckets[fi]['y'].append(y)
            buckets[fi]['c'].append(c)

        out = []
        for fi in sorted(buckets.keys()):
            xs = buckets[fi]['x']
            ys = buckets[fi]['y']
            cs = buckets[fi]['c']
            out.append((fi, sum(xs)/len(xs), sum(ys)/len(ys), sum(cs)/len(cs)))
        return out

    # simple enumeration
    return [(i, x, y, c) for i, (_,x,y,c) in enumerate(samples)]

def convert_gaze_file(input_path: str, fps: Optional[float] = None) -> str:
    """
    Convert a JSON or CSV gaze file to the standardized frame,x,y,confidence format.
    Returns path to the converted CSV file.
    """
    input_lower = input_path.lower()
    
    if input_lower.endswith(".json"):
        samples = read_pupil_json(input_path)
        logger.info(f"Loaded {len(samples)} samples from JSON file")
    elif input_lower.endswith(".csv"):
        samples = read_tobii_csv(input_path)
        logger.info(f"Loaded {len(samples)} samples from CSV file")
    else:
        raise ValueError("Input must be .json (Pupil) or .csv (Tobii)")

    rows = align_to_frames(samples, fps)
    
    # Create temporary converted file
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "converted_gaze.csv")
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame","x","y","confidence"])
        for fi,x,y,c in rows:
            w.writerow([fi, x, y, c])
    
    logger.info(f"Converted {len(rows)} gaze data rows to standardized format")
    return output_path

# --- main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert Pupil Neon JSON or Tobii Pro Glasses 3 CSV to frame,x,y,confidence CSV.")
    ap.add_argument("input", help="Path to input .json or .csv")
    ap.add_argument("output", help="Path to output CSV")
    ap.add_argument("--fps", type=float, default=None, help="Video FPS to align by timestamp (optional)")
    args = ap.parse_args()

    in_lower = args.input.lower()
    if in_lower.endswith(".json"):
        samples = read_pupil_json(args.input)
    elif in_lower.endswith(".csv"):
        samples = read_tobii_csv(args.input)
    else:
        print("Input must be .json (Pupil) or .csv (Tobii).", file=sys.stderr)
        sys.exit(2)

    rows = align_to_frames(samples, args.fps)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame","x","y","confidence"])
        for fi,x,y,c in rows:
            w.writerow([fi, x, y, c])

    print(f"Wrote {len(rows)} rows to {args.output}")

if __name__ == "__main__":
    main()