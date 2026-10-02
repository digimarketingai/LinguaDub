import os, gc, glob, ctypes, sys, site, subprocess, tempfile, traceback, time, re, html
os.environ["COQUI_TOS_AGREED"] = "1"

import numpy as np
import soundfile as sf
import torch
import requests
import gradio as gr
import transformers
from deep_translator import GoogleTranslator

if int(transformers.__version__.split(".")[0]) >= 5:
    print(f"WARNING: transformers {transformers.__version__} is loaded. "
          "Install transformers<5, restart the session, then run this cell.")

APP_NAME = "LinguaDub Studio"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 24000  # XTTS output sample rate

# name: (XTTS code, Google/MyMemory code, NLLB code)
LANGS = {
    "English":    ("en",    "en",    "eng_Latn"),
    "Spanish":    ("es",    "es",    "spa_Latn"),
    "French":     ("fr",    "fr",    "fra_Latn"),
    "German":     ("de",    "de",    "deu_Latn"),
    "Italian":    ("it",    "it",    "ita_Latn"),
    "Portuguese": ("pt",    "pt",    "por_Latn"),
    "Polish":     ("pl",    "pl",    "pol_Latn"),
    "Turkish":    ("tr",    "tr",    "tur_Latn"),
    "Russian":    ("ru",    "ru",    "rus_Cyrl"),
    "Dutch":      ("nl",    "nl",    "nld_Latn"),
    "Czech":      ("cs",    "cs",    "ces_Latn"),
    "Arabic":     ("ar",    "ar",    "arb_Arab"),
    "Chinese":    ("zh-cn", "zh-CN", "zho_Hans"),
    "Hungarian":  ("hu",    "hu",    "hun_Latn"),
    "Korean":     ("ko",    "ko",    "kor_Hang"),
    "Japanese":   ("ja",    "ja",    "jpn_Jpan"),
    "Hindi":      ("hi",    "hi",    "hin_Deva"),
}

# Rough speaking rate (characters per second) used to ESTIMATE how long a text
# takes to speak, until the real rate of the cloned voice has been measured.
DEFAULT_CPS = {
    "en": 15, "es": 15, "fr": 15, "de": 14, "it": 15, "pt": 15, "pl": 13,
    "tr": 13, "ru": 13, "nl": 14, "cs": 13, "ar": 12, "zh-cn": 4.5,
    "hu": 12, "ko": 6, "ja": 7.5, "hi": 13,
}

# Whisper-detected language code -> NLLB code (source language for NLLB)
WHISPER_TO_NLLB = {
    "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
    "it": "ita_Latn", "pt": "por_Latn", "pl": "pol_Latn", "tr": "tur_Latn",
    "ru": "rus_Cyrl", "nl": "nld_Latn", "cs": "ces_Latn", "ar": "arb_Arab",
    "zh": "zho_Hans", "hu": "hun_Latn", "ko": "kor_Hang", "ja": "jpn_Jpan",
    "hi": "hin_Deva", "uk": "ukr_Cyrl", "sv": "swe_Latn", "da": "dan_Latn",
    "fi": "fin_Latn", "no": "nob_Latn", "ro": "ron_Latn", "el": "ell_Grek",
    "bg": "bul_Cyrl", "he": "heb_Hebr", "fa": "pes_Arab", "id": "ind_Latn",
    "vi": "vie_Latn", "th": "tha_Thai", "ca": "cat_Latn", "sk": "slk_Latn",
    "hr": "hrv_Latn", "sr": "srp_Cyrl", "bn": "ben_Beng", "ta": "tam_Taml",
    "ur": "urd_Arab", "ms": "zsm_Latn",
}

ENGINES = {
    "MyMemory (online, simple API)":  "mymemory",
    "NLLB (offline)":                 "nllb",
    "Google (online, often blocked)": "google",
}
FALLBACK_ORDER = ["mymemory", "nllb", "google"]

FIT_MODES = {
    "Match slot: speed up AND slow down (most accurate)": "match",
    "Shorten only: speed up long lines":                  "shorten",
    "Off: natural speed":                                 "off",
}

# ---------------------------------------------------------------- helpers
def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr[-800:]}")
    return r.stdout

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def preload_cuda12_libs():
    roots = set(sys.path + site.getsitepackages())
    def load(pattern):
        for r in roots:
            for f in sorted(glob.glob(os.path.join(r, pattern))):
                try:
                    ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
    load("nvidia/cublas/lib/libcublasLt.so.12*")
    load("nvidia/cublas/lib/libcublas.so.12*")
    load("nvidia/cudnn/lib/libcudnn*.so.9*")

if DEVICE == "cuda":
    try:
        preload_cuda12_libs()
    except Exception as e:
        print("CUDA12 preload skipped:", e)

# ---------------------------------------------------------------- Whisper
_whisper = {}

def get_whisper(name, cpu=False):
    key = (name, cpu)
    if key in _whisper:
        return _whisper[key]
    _whisper.clear(); free_gpu()
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    if cpu or DEVICE != "cuda":
        m = WhisperModel(name, device="cpu", compute_type="int8")
    else:
        try:
            m = WhisperModel(name, device="cuda", compute_type="float16")
        except Exception as e:
            print("Whisper GPU load failed, using CPU int8:", e)
            m = WhisperModel(name, device="cpu", compute_type="int8")
            key = (name, True)
    _whisper[key] = BatchedInferencePipeline(model=m)
    return _whisper[key]

def transcribe_safe(audio16, name):
    def go(cpu):
        pipe = get_whisper(name, cpu=cpu)
        it, info = pipe.transcribe(
            audio16, batch_size=8 if (DEVICE == "cuda" and not cpu) else 1)
        raw = [dict(start=s.start, end=s.end, text=s.text.strip())
               for s in it if s.text.strip()]
        return raw, info
    try:
        return go(cpu=(DEVICE != "cuda"))
    except RuntimeError as e:
        msg = str(e).lower()
        if DEVICE == "cuda" and any(k in msg for k in ("libcu", "cublas", "cudnn", "cuda")):
            print("GPU Whisper failed (", e, ") -> retrying on CPU (slower)")
            gr.Warning("GPU Whisper unavailable, using CPU. Transcription will be slower.")
            return go(cpu=True)
        raise

# ---------------------------------------------------------------- Translation engines
# Each engine takes (texts, src_whisper_code, tgt_name, want) and returns, for every
# text, a LIST of candidate translations ordered best-first (empty list = failed).
# "want" is how many candidates we would like (1 when length control is off).

# ---- MyMemory (plain HTTP API, no key; ~5000 chars/day anonymous)
def _mm_code(c):
    return "zh-CN" if c.lower() in ("zh", "zh-cn") else c

def _chunks(text, limit):
    parts = [p for p in re.split(r"(?<=[.!?。！？])\s*", text) if p.strip()]
    pieces = []
    for p in parts:
        while len(p) > limit:
            pieces.append(p[:limit]); p = p[limit:]
        if p:
            pieces.append(p)
    out, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) + 1 > limit:
            out.append(cur); cur = p
        else:
            cur = (cur + " " + p).strip()
    if cur:
        out.append(cur)
    return out or [text[:limit]]

def tr_mymemory(texts, src, tgt_name, want=1):
    pair = f"{_mm_code(src)}|{_mm_code(LANGS[tgt_name][1])}"
    out, quota_hit = [], False
    for t in texts:
        if quota_hit:
            out.append([]); continue
        pieces, extra = [], []
        limit = 450 if t.isascii() else 150  # MyMemory limit is in bytes
        chunks = _chunks(t, limit)
        for c in chunks:
            try:
                r = requests.get("https://api.mymemory.translated.net/get",
                                 params={"q": c, "langpair": pair}, timeout=20)
                j = r.json()
                txt = (j.get("responseData") or {}).get("translatedText")
                status = str(j.get("responseStatus"))
                if status == "429" or (txt and "MYMEMORY WARNING" in txt.upper()):
                    print("MyMemory daily quota reached.")
                    quota_hit = True; pieces = None; break
                if status != "200" or not txt:
                    pieces = None; break
                pieces.append(html.unescape(txt))
                if want > 1 and len(chunks) == 1:
                    # extra candidates: close translation-memory matches
                    ms = sorted(j.get("matches") or [],
                                key=lambda m: -float(m.get("match", 0) or 0))
                    for m in ms:
                        if float(m.get("match", 0) or 0) >= 0.75 and m.get("translation"):
                            extra.append(html.unescape(m["translation"]))
            except Exception as e:
                print("MyMemory error:", str(e)[:100])
                pieces = None; break
            time.sleep(0.2)
        if pieces:
            out.append([" ".join(pieces)] + extra)
        else:
            out.append([])
    return out

# ---- NLLB-200 (local, no rate limits; first use downloads ~2.4 GB)
NLLB_NAME = "facebook/nllb-200-distilled-600M"
_nllb = None

def get_nllb():
    global _nllb
    if _nllb is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tok = AutoTokenizer.from_pretrained(NLLB_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_NAME)
        if DEVICE == "cuda":
            model = model.half()
        model = model.to(DEVICE).eval()
        _nllb = (tok, model)
    return _nllb

def tr_nllb(texts, src, tgt_name, want=1, bs=8):
    n_src = WHISPER_TO_NLLB.get(src)
    if n_src is None:
        raise RuntimeError(f"source language '{src}' not in NLLB mapping")
    n_tgt = LANGS[tgt_name][2]
    tok, model = get_nllb()
    tok.src_lang = n_src
    forced = tok.convert_tokens_to_ids(n_tgt)
    out = [[] for _ in texts]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to(model.device)
        nret = max(want, 1)
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=forced, max_new_tokens=256,
                                 num_beams=max(4, nret + 1), num_return_sequences=nret)
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        for j, i in enumerate(idx):
            out[i] = [d.strip() for d in dec[j * nret:(j + 1) * nret] if d.strip()]
        if want > 1:
            # second pass that prefers SHORTER output (negative length penalty)
            try:
                with torch.no_grad():
                    gen = model.generate(**enc, forced_bos_token_id=forced, max_new_tokens=256,
                                         num_beams=5, num_return_sequences=2,
                                         length_penalty=-2.0)
                dec = tok.batch_decode(gen, skip_special_tokens=True)
                for j, i in enumerate(idx):
                    out[i] += [d.strip() for d in dec[j * 2:(j + 1) * 2] if d.strip()]
            except Exception as e:
                print("NLLB short pass skipped:", str(e)[:100])
    return out

# ---- Google via deep_translator (online; often blocked on Colab)
def tr_google(texts, src, tgt_name, want=1):
    translator = GoogleTranslator(source="auto", target=LANGS[tgt_name][1])
    out, blocked = [], 0
    for t in texts:
        res = None
        if blocked < 3:
            for attempt in range(2):
                try:
                    res = translator.translate(t)
                    if res:
                        blocked = 0
                        break
                except Exception as e:
                    print(f"google attempt {attempt+1} failed: {str(e)[:80]}")
                time.sleep(2 * (attempt + 1))
            if not res:
                blocked += 1
        out.append([res] if res else [])
        time.sleep(0.3)
    return out

ENGINE_FUNCS = {"mymemory": tr_mymemory, "nllb": tr_nllb, "google": tr_google}

def translate_all(texts, src, tgt_name, engine_key, limits=None, est=None):
    """Translate with candidate selection.
    limits: per text (comfortable_seconds, max_seconds) or None (no length control)
    est:    function text -> estimated speaking seconds
    Returns (final_texts, n_failed, engines_used, candidate_lists, n_still_long)."""
    texts = [t.replace("\n", " ").strip() for t in texts]
    n = len(texts)
    cands = [[] for _ in texts]
    used = []
    want = 5 if limits else 1

    def add(i, lst):
        for c in lst or []:
            c = (c or "").strip()
            if c and c not in cands[i]:
                cands[i].append(c)

    def pick(i):
        cs = cands[i]
        if not cs:
            return None
        if limits is None:
            return cs[0]
        comfort, budget = limits[i]
        for c in cs:                       # best quality that fits comfortably
            if est(c) <= comfort:
                return c
        for c in cs:                       # best quality that fits with a speed-up
            if est(c) <= budget:
                return c
        return min(cs, key=est)            # nothing fits: shortest one

    def ok(i):
        if not cands[i]:
            return False
        return limits is None or est(pick(i)) <= limits[i][1]

    chain = [engine_key] + [e for e in FALLBACK_ORDER if e != engine_key]
    for key in chain:
        if all(ok(i) for i in range(n)):
            break
        if limits is None or key == "google":   # Google: only to fill missing translations
            pending = [i for i in range(n) if not cands[i]]
        else:
            pending = [i for i in range(n) if not ok(i)]
        if not pending:
            continue
        try:
            part = ENGINE_FUNCS[key]([texts[i] for i in pending], src, tgt_name, want)
            got = 0
            for i, r in zip(pending, part):
                before = len(cands[i])
                add(i, r)
                if len(cands[i]) > before:
                    got += 1
            if got:
                used.append(key)
            if got < len(pending):
                print(f"[{key}] produced output for {got}/{len(pending)}; trying next engine")
        except Exception as e:
            print(f"[{key}] failed: {type(e).__name__}: {str(e)[:150]}")
            if key == engine_key:
                gr.Warning(f"{key} failed ({type(e).__name__}); trying other engines.")
        free_gpu()

    failed = sum(1 for c in cands if not c)
    final = [pick(i) if cands[i] else texts[i] for i in range(n)]
    still_long = 0
    if limits is not None:
        still_long = sum(1 for i in range(n) if cands[i] and est(final[i]) > limits[i][1])
    return final, failed, (" + ".join(used) if used else "none"), cands, still_long

# ---------------------------------------------------------------- XTTS
_tts = None
def get_tts():
    global _tts
    if _tts is None:
        # Shim for transformers >= 5.1, which removed a helper coqui-tts still imports.
        import transformers.pytorch_utils as pu
        if not hasattr(pu, "isin_mps_friendly"):
            pu.isin_mps_friendly = lambda elements, test_elements: torch.isin(elements, test_elements)
        from TTS.api import TTS
        _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
    return _tts

# ---------------------------------------------------------------- speaking-time estimate
def estimate_dur(text, st):
    """Estimated natural speaking time (s). Uses the measured rate of the cloned
    voice once we have enough samples, otherwise a per-language default."""
    vals = list(st.get("cps_map", {}).values())
    if len(vals) >= 3:
        cps = float(np.median(vals))
    else:
        cps = DEFAULT_CPS.get(st["xtts"], 13.0)
    return max(len(text.strip()), 1) / max(cps, 1.0)

def record_rate(st, text, dur):
    if len(text) >= 8 and dur > 0.4:
        cm = st["cps_map"]
        cm.pop(text, None)
        cm[text] = len(text) / dur
        while len(cm) > 200:
            del cm[next(iter(cm))]

# ---------------------------------------------------------------- line timing control
def trim_silence(wav, thr=0.01, pad=0.04):
    """Remove leading/trailing silence so the measured length is the real speech length."""
    if len(wav) == 0:
        return wav
    peak = float(np.max(np.abs(wav)))
    if peak < 1e-4:
        return wav
    idx = np.where(np.abs(wav) > peak * thr)[0]
    if len(idx) == 0:
        return wav
    a = max(int(idx[0]) - int(pad * SR), 0)
    b = min(int(idx[-1]) + int(pad * SR), len(wav))
    return wav[a:b]

def apply_tempo(wav, speed):
    """Time-stretch with ffmpeg atempo (keeps pitch). speed must be in [0.5, 2.0]."""
    speed = float(np.clip(speed, 0.5, 2.0))
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, "a.wav"), os.path.join(d, "b.wav")
        sf.write(a, wav, SR)
        sh(["ffmpeg", "-y", "-i", a, "-filter:a", f"atempo={speed:.4f}", b])
        out, _ = sf.read(b, dtype="float32")
    return out

def fade_out(w, ms=40):
    n = min(int(SR * ms / 1000), len(w))
    if n > 0:
        w = w.copy()
        w[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return w

def fit_line(wav, slot, avail, mode, min_sp, max_sp, manual):
    """Fit one line of speech into its time slot.
    slot  = the Start->End window the user sees in the table
    avail = how long the line may run at most (slot + borrowed silence gap)
    Returns (wav, speed_used, notes)."""
    dur = len(wav) / SR
    notes = []
    sp = 1.0
    forced = False

    if manual and manual > 0:                       # user-specified speed for this line
        sp = float(np.clip(manual, 0.5, 2.0))
        notes.append("manual")
    elif mode != "off":
        if dur > slot:
            sp = min(dur / slot, max_sp)            # speed up long lines
        elif mode == "match" and dur < slot:
            sp = max(dur / slot, min_sp)            # slow down short lines
        if dur / sp > avail + 0.02:                 # still too long -> emergency speed-up
            need = min(dur / avail, 2.0)
            if need > sp:
                sp = need
                forced = True

    if abs(sp - 1.0) >= 0.02:
        wav = apply_tempo(wav, sp)
    else:
        sp = 1.0

    if forced:
        notes.append("forced")
    elif not notes:
        if sp > 1.0:
            notes.append("sped up")
        elif sp < 1.0:
            notes.append("slowed")

    if mode != "off" and not (manual and manual > 0):
        limit = int(avail * SR)
        if len(wav) > limit + int(0.02 * SR):       # still overruns -> cut with a fade
            wav = fade_out(wav[:limit])
            notes.append("truncated")

    return wav, sp, (", ".join(notes) if notes else "ok")

# ---------------------------------------------------------------- script table helpers
HEADERS = ["#", "Start (s)", "End (s)", "Speed (x, 0=auto)", "Original script", "Translated script"]
REPORT_HEADERS = ["#", "Slot (s)", "Natural (s)", "Final (s)", "Speed (x)", "Note"]

def _clean(v):
    if v is None:
        return ""
    if isinstance(v, float) and v != v:  # NaN
        return ""
    return str(v).strip()

def _num(v):
    try:
        return float(_clean(v))
    except ValueError:
        return None

def parse_table(table):
    """Editable table -> list of row dicts sorted by start time."""
    if hasattr(table, "values"):          # pandas DataFrame
        table = table.values.tolist()
    rows = []
    for r in table or []:
        if len(r) < 6:
            continue
        text, tr = _clean(r[4]), _clean(r[5])
        if not text and not tr:
            continue
        st, en = _num(r[1]), _num(r[2])
        if st is None or en is None:
            continue
        if en <= st:
            en = st + 0.5
        speed = _num(r[3]) or 0.0
        rid = _num(r[0])
        rows.append(dict(id=int(rid) if rid is not None else None,
                         start=max(st, 0.0), end=en, speed=max(speed, 0.0),
                         text=text, tr=tr))
    rows.sort(key=lambda x: x["start"])
    return rows

def commit(rows, st):
    """Renumber rows, remember them as the 'last known' script, return table + SRTs."""
    st["orig_map"], st["tr_map"] = {}, {}
    table = []
    for i, r in enumerate(rows, 1):
        r["id"] = i
        st["orig_map"][i], st["tr_map"][i] = r["text"], r["tr"]
        table.append([i, round(r["start"], 2), round(r["end"], 2),
                      round(r.get("speed", 0.0), 2), r["text"], r["tr"]])
    return table, write_srts(rows, st)

def _srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def write_srts(rows, st):
    paths = []
    for key, name in (("text", "original"), ("tr", "translated")):
        p = os.path.join(st["work"], f"{name}_v{st['ver']}.srt")
        with open(p, "w", encoding="utf-8") as f:
            for i, r in enumerate(rows, 1):
                f.write(f"{i}\n{_srt_time(r['start'])} --> {_srt_time(r['end'])}\n{r[key]}\n\n")
        paths.append(p)
    return paths

def needs_translation(r, st):
    """True if a row's translation is empty, or its original was edited while
    its translation was left untouched."""
    if not r["text"]:
        return False
    if not r["tr"]:
        return True
    rid = r["id"]
    if rid in st["orig_map"]:
        return r["text"] != st["orig_map"][rid] and r["tr"] == st["tr_map"].get(rid)
    return False  # new row that already has a translation typed in

def translate_rows(rows, st, engine_key, fit_on=True, max_sp=1.4):
    """Translate rows that need it, choosing wording that fits each row's time slot.
    Returns (n_failed, engines_used, n_still_too_long)."""
    idx = [i for i, r in enumerate(rows) if needs_translation(r, st)]
    if not idx:
        return 0, "none", 0
    if st["src"] == st["xtts"].split("-")[0]:
        for i in idx:
            rows[i]["tr"] = rows[i]["text"]
        return 0, "none", 0
    limits = None
    if fit_on:
        limits = []
        for i in idx:
            slot = max(rows[i]["end"] - rows[i]["start"], 0.3)
            limits.append((slot * min(max_sp, 1.2), slot * max_sp * 0.95))
    new, failed, used, cands, n_long = translate_all(
        [rows[i]["text"] for i in idx], st["src"], st["tgt"], engine_key,
        limits=limits, est=lambda t: estimate_dur(t, st))
    for i, t, c in zip(idx, new, cands):
        rows[i]["tr"] = t
        st["cands"][rows[i]["text"]] = c      # kept so the safety check can swap later
    return failed, used, n_long

def estimate_report(rows, st, max_sp):
    """Quick estimated timing report (no voice synthesis)."""
    rep, too_long = [], 0
    for n, r in enumerate(rows, 1):
        if not r["tr"]:
            continue
        slot = max(r["end"] - r["start"], 0.3)
        e = estimate_dur(r["tr"], st)
        need = e / slot
        sp = min(max(need, 1.0), max_sp) if need > 1.02 else 1.0
        if need > max_sp * 1.02:
            note = "too long (est.) - shorten or widen End"
            too_long += 1
        elif need > 1.02:
            note = "will speed up (est.)"
        else:
            note = "ok (est.)"
        rep.append([n, round(slot, 2), round(e, 2), round(e / sp, 2), round(sp, 2), note])
    return rep, too_long

# ---------------------------------------------------------------- synth + render
def shorter_alternative(s, cur_dur, slot, max_sp, st, tried):
    """Next-best shorter candidate for a machine-made translation, or None.
    Translations the user wrote or edited themselves are never replaced."""
    cands = st["cands"].get(s["text"], [])
    if s["tr"] not in cands:
        return None
    ratio = cur_dur / max(estimate_dur(s["tr"], st), 0.1)   # calibrate estimate to this line
    pool = [(c, estimate_dur(c, st) * ratio) for c in cands if c not in tried]
    pool = [(c, p) for c, p in pool if p < cur_dur * 0.95]
    if not pool:
        return None
    target = slot * max_sp * 0.95
    for c, p in pool:                       # best quality that should fit
        if p <= target:
            return c
    return min(pool, key=lambda x: x[1])[0]

def synth_and_render(rows, st, keep_orig, fit_mode, min_sp, max_sp, borrow, auto_fit, progress):
    tts = get_tts()
    code, ref, cache = st["xtts"], st["ref"], st["cache"]
    segs = [(n, r) for n, r in enumerate(rows, 1) if r["tr"]]
    if not segs:
        raise gr.Error("The translated script is empty.")
    vid_dur = st["dur"]
    buf = np.zeros(int((vid_dur + 30) * SR), dtype=np.float32)
    cursor, reused = 0.0, 0
    report = []

    def get_wav(text):
        w = cache.get((text, code))
        if w is not None:
            return w, True
        try:
            w = np.asarray(tts.tts(text=text, speaker_wav=ref, language=code), dtype=np.float32)
        except Exception as e:
            print(f"TTS failed: {e}")
            return None, False
        w = trim_silence(w)
        cache[(text, code)] = w             # cache the raw (unfitted) speech
        return w, False

    for i, (n, s) in enumerate(segs):
        progress(0.45 + 0.45 * i / len(segs), desc=f"Synthesizing {i+1}/{len(segs)}")
        start = max(s["start"], cursor)
        slot = max(s["end"] - start, 0.3)
        nxt = segs[i + 1][1]["start"] if i + 1 < len(segs) else s["end"] + 1.5
        avail = max(nxt - start - 0.08, slot) if borrow else slot

        wav, was_cached = get_wav(s["tr"])
        if wav is None:
            report.append([n, round(slot, 2), 0, 0, 0, "TTS failed"])
            continue
        reused += was_cached

        # safety check: real audio still too long even at max speed -> try shorter wording
        swapped = False
        if auto_fit and fit_mode != "off" and not s.get("speed"):
            tried = {s["tr"]}
            for _ in range(3):
                dur = len(wav) / SR
                if dur <= slot * max_sp * 1.02:
                    break
                alt = shorter_alternative(s, dur, slot, max_sp, st, tried)
                if not alt:
                    break
                tried.add(alt)
                progress(0.45 + 0.45 * i / len(segs), desc=f"Shortening line {n}")
                w2, _c = get_wav(alt)
                if w2 is not None and len(w2) < len(wav):
                    s["tr"], wav, swapped = alt, w2, True

        record_rate(st, s["tr"], len(wav) / SR)

        w, sp, note = fit_line(wav, slot, avail, fit_mode, min_sp, max_sp, s.get("speed", 0.0))
        if swapped:
            note += ", shorter text"
        if start > s["start"] + 0.05:
            note += f", shifted +{start - s['start']:.2f}s"
        report.append([n, round(slot, 2), round(len(wav) / SR, 2),
                       round(len(w) / SR, 2), round(sp, 2), note])

        a = int(start * SR); b = min(a + len(w), len(buf))
        if b > a:
            buf[a:b] += w[: b - a]
        cursor = start + len(w) / SR
    print(f"Reused {reused}/{len(segs)} cached segments.")

    progress(0.93, desc="Rendering video")
    buf = buf[: int(vid_dur * SR)]
    if keep_orig > 0:
        orig24, _ = sf.read(st["a24"], dtype="float32")
        n_ = min(len(buf), len(orig24))
        buf[:n_] += keep_orig * orig24[:n_]
    peak = float(np.max(np.abs(buf))) or 1.0
    buf = buf / peak * 0.95
    st["ver"] += 1
    dub_wav = os.path.join(st["work"], f"dub_v{st['ver']}.wav")
    sf.write(dub_wav, buf, SR)
    out = os.path.join(st["work"], f"dubbed_v{st['ver']}.mp4")
    sh(["ffmpeg", "-y", "-i", st["video"], "-i", dub_wav, "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", f"{vid_dur:.3f}", out])
    return out, report

def fit_summary(report):
    short = sum(1 for r in report if "shorter text" in r[5])
    bad = [r for r in report if any(k in r[5] for k in ("forced", "truncated", "shifted", "failed"))]
    msg = ""
    if short:
        msg += f"✂️ {short} translation(s) were automatically shortened to fit. "
    if not bad:
        return msg + "All lines fit their time slots."
    ids = ", ".join(str(r[0]) for r in bad[:15])
    return msg + (f"⚠️ {len(bad)} line(s) are tight or overrun (#{ids}). "
                  "Shorten their translation, widen their End time, or set a manual speed.")

# ---------------------------------------------------------------- step 1: first dub
def dub(video, target_lang, engine_label, whisper_name, keep_orig,
        fit_label, min_sp, max_sp, borrow, auto_fit, _old_state, progress=gr.Progress()):
    try:
        if not video:
            raise gr.Error("Upload a video first.")
        xtts_code = LANGS[target_lang][0]
        engine_key = ENGINES[engine_label]
        fit_mode = FIT_MODES[fit_label]
        work = tempfile.mkdtemp()

        progress(0.01, desc="Loading XTTS")
        get_tts()

        progress(0.04, desc="Extracting audio")
        a16, a24 = os.path.join(work, "a16.wav"), os.path.join(work, "a24.wav")
        sh(["ffmpeg", "-y", "-i", video, "-vn", "-ac", "1", "-ar", "16000", a16])
        sh(["ffmpeg", "-y", "-i", video, "-vn", "-ac", "1", "-ar", str(SR), a24])
        audio16, _ = sf.read(a16, dtype="float32")
        orig24, _ = sf.read(a24, dtype="float32")
        vid_dur = float(sh(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=nw=1:nk=1", video]).strip())

        progress(0.10, desc="Transcribing (Whisper)")
        raw, info = transcribe_safe(audio16, whisper_name)
        if not raw:
            raise gr.Error("No speech detected in this video.")
        _whisper.clear(); free_gpu()

        # reference voice (longest clips, up to ~15 s)
        progress(0.28, desc="Preparing voice reference")
        ordered = sorted(raw, key=lambda s: s["end"] - s["start"], reverse=True)
        parts, total = [], 0.0
        for s in ordered:
            clip = orig24[int(s["start"] * SR): int(s["end"] * SR)]
            if len(clip) < SR // 2:
                continue
            parts.append(clip); total += len(clip) / SR
            if total >= 15:
                break
        if not parts:
            parts = [orig24[: SR * 10]]
        ref = os.path.join(work, "ref.wav")
        sf.write(ref, np.concatenate(parts), SR)

        st = dict(work=work, video=video, a24=a24, ref=ref, dur=vid_dur,
                  src=info.language, tgt=target_lang, xtts=xtts_code,
                  orig_map={}, tr_map={}, cache={}, ver=0,
                  cands={}, cps_map={})

        rows = [dict(id=None, start=s["start"], end=s["end"], speed=0.0,
                     text=s["text"], tr="") for s in raw]

        progress(0.32, desc="Translating (fitting to time slots)")
        if st["src"] == xtts_code.split("-")[0]:
            gr.Warning("Video is already in the target language; skipping translation.")
            for r in rows:
                r["tr"] = r["text"]
            failed, used, n_long = 0, "none", 0
        else:
            failed, used, n_long = translate_rows(rows, st, engine_key, auto_fit, max_sp)
        if failed:
            gr.Warning(f"{failed} segment(s) could not be translated and were left in the "
                       "original language. You can fix them in the script editor.")

        out, report = synth_and_render(rows, st, keep_orig, fit_mode, min_sp, max_sp,
                                       borrow, auto_fit, progress)
        table, srts = commit(rows, st)
        status = (f"✅ Version {st['ver']} ready. Detected language: **{info.language}** | "
                  f"Translation engine(s): **{used}**. {fit_summary(report)} "
                  "Edit the script below, then press **Redo dubbing**.")
        return out, table, st, status, srts, report

    except gr.Error:
        raise
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------- step 2a: re-translate edited rows
def retranslate(table, engine_label, auto_fit, max_sp, st):
    try:
        if not st or "work" not in st:
            raise gr.Error("Run “Dub it” first.")
        rows = parse_table(table)
        if not rows:
            raise gr.Error("The script table is empty.")
        failed, used, n_long = translate_rows(rows, st, ENGINES[engine_label], auto_fit, max_sp)
        table, srts = commit(rows, st)
        rep, too_long = estimate_report(rows, st, max_sp)
        msg = f"✅ Translations updated (engine(s): {used}). "
        if auto_fit:
            msg += "Wording was chosen to fit each time slot. "
        if too_long:
            bad = [str(r[0]) for r in rep if "too long" in r[5]][:15]
            msg += (f"⚠️ {too_long} line(s) still look too long (#{', '.join(bad)}): "
                    "shorten the wording or widen End. ")
        else:
            msg += "All lines are estimated to fit. "
        msg += "The report below shows estimates; press **Redo dubbing** for exact timing."
        if failed:
            msg += f" ⚠️ {failed} row(s) could not be translated."
        return table, st, msg, srts, rep
    except gr.Error:
        raise
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------- step 2b: redo dubbing
def redub(table, engine_label, keep_orig, fit_label, min_sp, max_sp, borrow, auto_fit, st,
          progress=gr.Progress()):
    try:
        if not st or "work" not in st:
            raise gr.Error("Run “Dub it” first.")
        rows = parse_table(table)
        if not rows:
            raise gr.Error("The script table is empty.")
        progress(0.02, desc="Checking edited rows")
        failed, used, n_long = translate_rows(rows, st, ENGINES[engine_label], auto_fit, max_sp)
        if failed:
            gr.Warning(f"{failed} edited row(s) could not be translated and were left as they are.")
        out, report = synth_and_render(rows, st, keep_orig, FIT_MODES[fit_label],
                                       min_sp, max_sp, borrow, auto_fit, progress)
        table, srts = commit(rows, st)
        status = (f"✅ Version {st['ver']} ready (re-translated with: {used}). "
                  f"Unchanged lines reused cached audio. {fit_summary(report)}")
        return out, table, st, status, srts, report
    except gr.Error:
        raise
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------- UI
DESCRIPTION = f"""
# 🎙️ {APP_NAME}
### AI Video Dubbing with Voice Cloning, Script Editing and Line Timing Control · AI 视频配音、脚本编辑与逐句时间控制工作室

**English**
{APP_NAME} turns a video into another language while keeping the original speaker's voice.
1. **Dub it:** it transcribes the speech (Whisper), translates it (MyMemory by default, with automatic fallback), and re-voices it with a clone of the speaker's voice (XTTS v2).
2. **Translations that fit:** for every line, the app compares several candidate translations and picks the best one whose spoken length fits the line's Start–End slot. If the real voice is still too long, it automatically tries a shorter wording. This also applies when you re-translate edited rows.
3. **Line timing control:** every dubbed line is trimmed of silence and time-stretched (pitch preserved) to fit its slot. Choose a fit mode, set the minimum and maximum speed, and optionally borrow the silent gap before the next line. You can also set an exact speed for any single line.
4. **Edit:** the original and translated scripts appear in an editable table. Fix mistakes, rewrite lines, adjust timings and speeds, and add or delete rows. A timing report shows how each line fitted (estimates after re-translating, exact values after dubbing).
5. **Redo dubbing:** it generates a new video from your edited script. Lines you didn't touch reuse their audio, so redoing is fast.

Tip: if you edit only the *original* text of a row, its translation is refreshed automatically and fitted to the slot. If you edit the *translation* yourself, your wording is kept exactly. Works best with a single speaker.

**中文**
{APP_NAME} 可将视频配音成另一种语言，并保留原说话人的声音。
1. **开始配音：** 自动识别语音（Whisper），翻译（默认 MyMemory，失败时自动切换备用引擎），再用克隆的原声合成配音（XTTS v2）。
2. **译文适配时长：** 对每一句，程序会比较多个候选译文，选出朗读时长能适配该句“开始–结束”时间段的最佳译文；如果实际配音仍然过长，会自动改用更短的译法。重新翻译已修改的行时同样适用。
3. **逐句时间控制：** 每一句配音都会先去除首尾静音，再在保持音高的前提下变速，以贴合其时间段。你可以选择适配模式、设置最小/最大语速，也可以借用下一句之前的空白间隙，还可以为任意一句单独指定语速。
4. **编辑脚本：** 原文和译文显示在可编辑表格中，你可以修改错误、改写句子、调整时间与语速，也可以增删行。时间报告会显示每一句的适配情况（重新翻译后为估算值，配音后为精确值）。
5. **重新配音：** 根据编辑后的脚本生成新的视频；未修改的句子会直接复用已生成的音频，速度更快。

提示：如果只修改某一行的*原文*，其译文会自动更新并适配时长；如果你亲自修改了*译文*，则完全保留你的措辞。最适合单人说话的视频。
"""

with gr.Blocks(title=APP_NAME) as demo:
    gr.Markdown(DESCRIPTION)
    state = gr.State({})

    with gr.Row():
        with gr.Column():
            vid = gr.Video(label="Input video / 输入视频")
            lang = gr.Dropdown(list(LANGS), value="Spanish", label="Target language / 目标语言")
            with gr.Accordion("⏱️ Line timing control / 逐句时间控制", open=True):
                fit = gr.Dropdown(list(FIT_MODES), value=list(FIT_MODES)[0],
                                  label="Fit mode / 适配模式")
                min_sp = gr.Slider(0.6, 1.0, value=0.85, step=0.05,
                                   label="Min speed (slow-down limit) / 最小语速")
                max_sp = gr.Slider(1.0, 2.0, value=1.4, step=0.05,
                                   label="Max speed (speed-up limit) / 最大语速")
                borrow = gr.Checkbox(value=True,
                                     label="Borrow the silent gap before the next line / 借用下一句前的空白间隙")
                auto_fit = gr.Checkbox(value=True,
                                       label="Make translations fit the time slot (picks shorter wording when needed) / 让译文适配时间段")
            with gr.Accordion("Advanced / 高级设置", open=False):
                eng = gr.Dropdown(list(ENGINES), value=list(ENGINES)[0],
                                  label="Translation engine (others are fallback) / 翻译引擎")
                wmodel = gr.Dropdown(["large-v3-turbo", "large-v3", "medium", "small", "base"],
                                     value="large-v3-turbo", label="Whisper model / 识别模型")
                keep = gr.Slider(0, 0.4, value=0.0, step=0.05,
                                 label="Keep original audio volume / 保留原声音量")
            btn = gr.Button("1. Dub it / 开始配音", variant="primary")
        with gr.Column():
            out_vid = gr.Video(label="Dubbed video / 配音后的视频")
            status = gr.Markdown("Upload a video and press **Dub it**.")

    gr.Markdown("## ✏️ Script editor / 脚本编辑\n"
                "Edit the **Original script** and/or **Translated script** cells, change times, "
                "set a per-line **Speed** (0 = automatic), or add/delete rows. Then press "
                "**Re-translate** (optional) and **Redo dubbing**. 编辑原文或译文，可修改时间、"
                "为单句设置语速（0 为自动）、增删行，然后点击“重新翻译”（可选）和“重新配音”。")
    table = gr.Dataframe(headers=HEADERS,
                         datatype=["number", "number", "number", "number", "str", "str"],
                         col_count=(6, "fixed"), row_count=(1, "dynamic"),
                         interactive=True, wrap=True, type="array",
                         label="Script / 脚本")
    with gr.Row():
        btn_tr = gr.Button("Re-translate edited rows (fits time) / 重新翻译已修改的行（适配时长）")
        btn_redo = gr.Button("2. Redo dubbing / 重新配音", variant="primary")

    gr.Markdown("## 📊 Timing report / 时间报告\n"
                "Slot = your Start→End window. Natural = speech length before fitting. "
                "Final = length after fitting. After *Re-translate* these are estimates (est.); "
                "after dubbing they are exact. 时间段 = 开始→结束；自然时长 = 适配前；最终时长 = 适配后。"
                "重新翻译后为估算值，配音后为精确值。")
    report_tbl = gr.Dataframe(headers=REPORT_HEADERS, interactive=False, wrap=True,
                              label="How each line fitted / 每句适配情况")
    files = gr.File(label="Download scripts (.srt) / 下载字幕", file_count="multiple")

    btn.click(dub, [vid, lang, eng, wmodel, keep, fit, min_sp, max_sp, borrow, auto_fit, state],
              [out_vid, table, state, status, files, report_tbl])
    btn_tr.click(retranslate, [table, eng, auto_fit, max_sp, state],
                 [table, state, status, files, report_tbl])
    btn_redo.click(redub, [table, eng, keep, fit, min_sp, max_sp, borrow, auto_fit, state],
                   [out_vid, table, state, status, files, report_tbl])

demo.queue().launch(share=True, debug=True)
