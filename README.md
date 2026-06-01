# Qpic

FastAPI app that accepts a PDF, detects MCQ question regions using a smart
3-tier pipeline (text → OCR → AI fallback), crops/stitches each question
(including cross-page questions), and returns a ZIP of images. It ships with a
polished single-page web UI (Adobe Acrobat-style, with light/dark/system
themes) and can run either in the browser or as a native desktop app.

**Highlights**

- **Smart detection** — text, OCR and an optional AI vision tier, with manual
  review + hand-fixing before download.
- **Acrobat-style UI** — a clean web front-end served from `static/index.html`;
  a top app bar, tool tabs (Auto Crop / Manual Crop / Rename Batch / Tools) and a review canvas.
- **Two desktop backends** — pywebview (small) or Qt/PySide6 (consistent
  Chromium rendering). See *Desktop app* below.
- **Offline-capable** — bundles Tesseract for OCR; AI is only used when a key is
  configured and Online mode is on.

## Smart mode + manual review

Beyond the one-shot `/crop` flow, the app has a **Smart mode** (on by default in
the UI) that handles *any* PDF layout and lets you fix detection by hand before
downloading:

1. **Analyze** (`POST /api/analyze`) — runs the pipeline in *smart* mode. The
   cheap text/OCR tiers are accepted only when they look confident (good
   density + unbroken numbering); otherwise it escalates to the Claude vision
   tier so odd layouts/numbering still get detected. Returns the detected
   regions, per-page geometry + preview images, and **review notes**, including
   crops that look **cut off / half** (a crop that stops at the page edge, or is
   much shorter than its neighbours and probably lost its options), likely
   duplicates, and numbering gaps.
2. **Review popup** — the UI shows every detected box over the page previews and
   flags anything uncertain. For a cut-off item you hit **Re-select** (or the
   **Fix** button on its note) and drag the *correct full region* on the page —
   the item's box is replaced. You can also **draw** a box for a missed item, or
   delete extra/duplicate ones. **Snap to content** (on by default) auto-tightens
   any box you draw to the actual text/figure inside it, so a rough drag becomes
   a clean, content-hugging crop.

   Several checks make the review smarter:
   - **Gap recovery** — if the detected numbering skips a value (e.g. 19 → 21),
     the pipeline re-reads the lines between the neighbours with OCR
     digit-confusion fixups (`20.` misread as `2O.` / `Z0.`) and re-inserts the
     missed question instead of silently dropping it.
   - **Answer-key cross-check** — if the paper carries an answer key (`1-B 2-A
     3-D …`), it lists every question number, so the tool knows exactly how many
     questions exist. Any number in the key that wasn't detected is reported as a
     high-confidence miss, and the key also drives gap recovery toward numbers a
     sequence alone wouldn't reveal (e.g. a question missing from the end).
   - **Option check** — standard MCQs have options (A)–(D). A crop that captured
     only some of them (e.g. just the left column `(A)/(C)` of a 2-up grid) is
     flagged as *likely missing its right-hand options* so you can re-select it.

### Scan quality + AI escalation

For scanned (image) PDFs the OCR tier now **deskews** a tilted page, denoises
speckle, and uses an adaptive (Otsu) threshold before reading — a small tilt
otherwise smears Tesseract's line grouping and drops question numbers. It also
records a **per-page confidence**; in Smart mode with an `ANTHROPIC_API_KEY`
configured, only the *low-confidence* pages are re-detected by the AI vision tier
and merged back in, so a few blurry pages get repaired without paying to send the
whole document to the model. Tune the cutoff with `OCR_MIN_CONFIDENCE` (default
75).

Stray horizontal dividers (question separators, table borders) drawn as flat
zero-thickness rules are now removed from crops, while fraction bars (`500/3`)
and text underlines are preserved as real content.
3. **Finalize** (`POST /api/finalize`) — combines the kept auto items and your
   corrected/hand-drawn ones into the final ZIP, re-rendered crisp from the PDF
   vector source. No re-upload: the source PDF is cached in the job dir by
   `/analyze`.

Turn Smart mode off to keep the original "type page ranges → straight to ZIP"
behaviour.

### Answer sheet (auto-generated)

When a paper carries an **answer key** (`1-B 2-A 3-D …`), every download also
includes an **answer sheet** that pairs each cropped question image with its
correct option:

- `answers.csv` — opens straight into Excel/Sheets (`file, question, answer`),
  e.g. `Q001.png, 1, B`.
- `answers.json` — the machine-readable form for importing into a quiz/Anki
  pipeline.

The sheet rides along in the **questions** and **combined** ZIPs (not the
solutions-only one, since it's keyed to the question images). The key is read
**for free** from a searchable PDF's text layer; on a **scanned** paper whose
text layer is empty, the AI vision tier (Opus) reads the key from the page
images instead — but only when **Online mode** is on and an AI key is
configured. A paper without any answer key simply ships no sheet.

Toggle it per run with the **Answer sheet** switch in the **Output options**
panel (on by default). To turn it off globally, set
`ANSWER_SHEET_ENABLED=false`.

### Online / Offline mode (AI vision)

The UI has an **Online mode (AI)** toggle (top of the "What to crop" panel):

- **On** — the AI vision tier is allowed. The cheap text/OCR tiers still run
  first; AI is only used as a fallback for hard layouts and for repairing
  low-confidence scanned pages.
- **Off** — a fully **offline** run. Only the text and OCR tiers are used, so no
  network calls are made. Use this when you have no key, no internet, or want
  guaranteed-local processing.

The toggle auto-disables itself when the server reports no AI key is configured.

**Configuring the AI key.** Two providers are supported (set in `.env`):

```bash
# OpenRouter (OpenAI-compatible — Gemini / Qwen / Llama / free models)
AI_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=nvidia/nemotron-nano-12b-v2-vl:free   # a free model, or a paid one for accuracy

# …or Anthropic
AI_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-opus-4-8
```

`AI_PROVIDER=auto` (default) prefers OpenRouter when its key is set, else
Anthropic. With no key configured, the app runs offline-only regardless of the
toggle.

### Question numbering style

Both `/crop` and `/analyze` accept `marker_style` to control which numbering is
treated as a real question, so sub-statements, option labels and equation
numbers in the body aren't mistaken for questions:

- `auto` (default) — prefer explicit `Q` markers; fall back to bare numbers.
- `q` — only `Q1` / `Q.1` / `Question 1` style markers count.
- `numbered` — only bare leading numbers (`1.`, `2)`) count.

The UI exposes this as a **Question numbering** dropdown. The chosen style is
honoured by every detection tier (text, OCR and the AI vision prompt).

## How to use the app

Once the server is running (see *How to run* below), open **http://localhost:8000** in your browser. Here's the typical workflow:

### 1. Upload a PDF
- Click **"Choose PDF"** (or drag-and-drop a file onto the upload area).
- Select your MCQ question paper PDF.

### 2. Configure detection options
| Option | What it does |
|---|---|
| **Question numbering** | `auto` works for most papers. Switch to `q` if questions are labelled `Q1/Q2…`, or `numbered` for bare `1. 2. 3.` style. |
| **Online mode (AI)** | Toggle ON to allow the AI vision fallback for tricky layouts. Requires an API key in `.env`. Toggle OFF for a fully offline run. |
| **Smart mode** | ON (default) — runs the full pipeline and opens the review canvas. OFF — skips review and goes straight to ZIP. |
| **Answer sheet** | ON (default) — bundles `answers.csv` + `answers.json` (each question image → correct option) when the PDF has an answer key. OFF — skips it. Lives in the **Output options** panel. |

### 3. Analyze
- Click **Analyze**. The app runs text → OCR → AI detection and shows a **review canvas** with every detected question box overlaid on the page previews.

### 4. Review & fix detections
- **Green boxes** = detected questions. Hover to see the question number.
- **Review notes** on the right flag anything suspicious (cut-off crops, numbering gaps, missing options).
- To fix a bad box: click its **Fix / Re-select** button, then drag the correct region on the page image.
- To add a missed question: click **Draw**, drag a box around it.
- To remove a duplicate: click the box → **Delete**.
- **Snap to content** (on by default) auto-tightens any box you draw to the actual text/figure inside it.

### 5. Download
- Once happy with the review, click **Finalize & Download**.
- Choose the download type:
  - **Combined** — questions + solutions in one ZIP (`QScombined.zip`)
  - **Questions only** — `Q.zip`
  - **Solutions only** — `S.zip`

### Rename Batch tab
Switch to the **Rename Batch** tab to bulk-rename a folder of already-cropped images using a custom prefix and numbering scheme.

### Tools tab (Compress / Edit / Preflight)

The **Tools** tab bundles three standalone PDF utilities, all powered by PyMuPDF
on the backend (no extra binaries beyond the optional Tesseract for OCR):

- **Compress PDF** — shrink a PDF by recompressing its images, subsetting fonts
  and cleaning the object streams. Pick a **level** (`light` / `balanced` /
  `strong` / `extreme`) or set a **target size in MB** and the tool pushes
  quality down until the file fits (best-effort, never below readable quality).
  The result reports the before/after size and the percentage saved.
- **Edit PDF** — edit text **in place**. Each text run is shown as a clickable
  box over a page preview; changes are re-inserted using the document's **own
  embedded font**, at the same size and colour, fitted to the original box (a
  Base-14 / closest-match fallback is used when the original font can't be
  reused). A **Run OCR** button turns a scanned/image PDF into a searchable one
  (invisible Tesseract text layer) so its text becomes editable too.
- **Preflight PDF** — a read-only prepress check: page count/sizes, embedded vs.
  non-embedded fonts, image resolution (low-DPI = blurry in print), RGB vs CMYK
  colour, encryption and a searchable-text check, rolled up into a single
  PASS / WARN / FAIL verdict with actionable messages.

---

## How to run

```bash
# 1. Install Tesseract (for OCR tier)
# Mac:   brew install tesseract
# Linux: sudo apt install tesseract-ocr
# Win:   https://github.com/UB-Mannheim/tesseract/wiki

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment (AI key optional)
cp .env.example .env
# Edit .env and optionally add ANTHROPIC_API_KEY

# 4. Start server
uvicorn app.main:app --reload --port 8000

# 5. Open browser
# http://localhost:8000

# 6. API docs
# http://localhost:8000/docs

# 7. Run tests
pytest tests/ -v

# 8. Docker (includes Tesseract automatically)
docker build -t qpic .
docker run -p 8000:8000 --env-file .env qpic
```

## Endpoints

- `POST /api/crop` — upload PDF (`multipart/form-data` field `file`) with optional query params `dpi`, `padding` and `marker_style`
- `POST /api/analyze` — smart-detect and return regions + page previews + review notes (no ZIP yet)
- `GET /api/analyze/{job_id}/page/{n}` — page-preview PNG for the manual-crop canvas
- `POST /api/snap` — tighten a roughly drawn box to the content inside it
- `POST /api/finalize` — JSON body of reviewed items (auto + manual) → builds the ZIPs
- `GET /api/crop/download/{job_id}` — download a ZIP. Optional `kind` query param: `combined` (default, questions + solutions → `QScombined.zip`), `questions` (questions only → `Q.zip`), or `solutions` (solutions only → `S.zip`). `question_prefix` / `solution_prefix` set the download filename.
- `GET /api/health` — health check

### Tools endpoints

- `POST /api/tools/compress` — `multipart/form-data` field `file`, plus `level` (`light`/`balanced`/`strong`/`extreme`) **or** `target_mb`. Returns sizes + `download_url`.
- `GET /api/tools/compress/download/{job_id}` — download the compressed PDF.
- `POST /api/tools/preflight` — field `file`; returns the full preflight report (verdict, checks, fonts, images, metadata).
- `POST /api/tools/edit/open` — field `file`; stages the PDF and returns editable text spans (with geometry/font/size/colour) + page previews.
- `POST /api/tools/edit/apply` — JSON `{job_id, edits:[{page, bbox, new_text, font?, size?, color?}]}`; applies font-matched replacements. Returns `download_url`.
- `POST /api/tools/edit/ocr` — field `file`, optional `languages` (e.g. `eng+hin`) and `dpi`; adds an invisible OCR text layer and makes the result the new editable source.
- `GET /api/tools/edit/download/{job_id}` — download the edited (or OCR'd) PDF.

## Desktop app (no terminal, no server to start manually)

The same app can be packaged as a native desktop app. The web server still runs,
but it's hidden inside the app and started/stopped automatically — you just
double-click an icon and a normal window opens.

```bash
# macOS / Linux
./build_desktop.sh
# -> dist/Qpic.app   (macOS)

# Windows (run in Command Prompt)
build_desktop.bat
# -> dist\Qpic\Qpic.exe
```

Notes:
- You can only build the macOS `.app` on a Mac and the Windows `.exe` on Windows;
  one machine can't build the other's binary. To build **both** without two
  machines, use the GitHub Actions workflow (see *CI builds* below).
- The build is **unsigned**, so the first launch on macOS shows an
  "unidentified developer" warning — right-click the app → **Open** to allow it.
- **OCR works offline out of the box.** The build vendors a self-contained
  Tesseract (binary + libraries + `eng`/`hin`/`osd` language data) into the app
  via `scripts/vendor_tesseract.py`, so scanned PDFs are handled with no
  separate Tesseract install on the user's machine. Any requested language the
  build host's Tesseract didn't ship (e.g. Hindi on the Windows installer) is
  downloaded automatically from the official `tessdata` repos at build time. At
  runtime the app finds it in this order: `TESSERACT_CMD` env var → the copy
  bundled inside the app → a standard system install → whatever is on `PATH`.
  The AI fallback still needs internet + an API key.
- Cropped images/zips are written to a per-user folder
  (`~/Library/Application Support/Qpic` on macOS).

### Qt (PySide6) variant

There are **two** desktop window backends; both run the same hidden FastAPI
server and show the same web UI, so feature-wise they're identical:

| | `desktop.py` (default) | `desktop_qt.py` (Qt) |
|---|---|---|
| Window | pywebview → OS webview (WKWebView / WebView2) | Qt `QWebEngineView` (bundled Chromium) |
| Rendering | depends on the OS webview | identical Chromium on every OS |
| Bundle size | smaller | larger (~150-200 MB, ships Chromium) |
| Build spec | `desktop.spec` | `desktop_qt.spec` |
| Build script | `build_desktop.sh` | `build_desktop_qt.sh` |

Use the Qt variant when you want pixel-identical rendering across macOS and
Windows and don't mind the larger download.

```bash
# Run from source
pip install -r requirements.txt -r requirements-desktop-qt.txt
python desktop_qt.py

# Build the bundle
./build_desktop_qt.sh        # -> dist/Qpic.app (macOS)
```


## CI builds (both macOS + Windows, no second machine)

`.github/workflows/build-desktop.yml` builds the desktop app on **both**
`macos-latest` and `windows-latest` runners. Each runner installs Tesseract,
vendors it into the bundle, runs PyInstaller, and uploads an installer-ready
archive (`Qpic-macOS.zip` / `Qpic-Windows.zip`).

- **Run it on demand:** Actions tab → *Build desktop apps* → *Run workflow*.
  Download the results from the run's *Artifacts*.
- **Cut a release:** push a tag like `v1.0.0` and the workflow attaches both
  archives to a GitHub Release automatically.

## License

Released under the [MIT License](LICENSE) — © 2026 Aniket Mishra.




















