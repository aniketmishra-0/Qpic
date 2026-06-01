/* ============================================================================
   Qpic full-screen PDF Editor — real, Acrobat-style editing.

   Talks to the same backend the inline Tools→Edit panel uses:
     POST /api/tools/edit/open           → stage a PDF, get spans + page geometry
     GET  /api/tools/edit/{job}/state    → re-open an already-staged job (no upload)
     GET  /api/tools/edit/{job}/page/{n} → page preview PNG
     POST /api/tools/edit/apply          → apply edits, returns a download URL
     POST /api/tools/edit/ocr            → add a searchable text layer to a scan

   Capabilities: edit existing text in place (original font matched on the
   server), add text boxes, insert images, add hyperlinks, white-out/erase,
   move/resize/delete added objects, zoom, multi-page scroll, open another PDF
   at any time, and save & download the edited PDF.
   ========================================================================== */
(function () {
  'use strict';

  const API = '/api/tools/edit';

  // ---- DOM refs ------------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const fileInput = $('fileInput');
  const imgInput = $('imgInput');
  const canvasScroll = $('canvasScroll');
  const thumbs = $('thumbs');
  const startScreen = $('startScreen');
  const dropzone = $('dropzone');
  const dzStatus = $('dzStatus');
  const toastEl = $('toast');

  const openBtn = $('openBtn');
  const ocrBtn = $('ocrBtn');
  const saveBtn = $('saveBtn');
  const saveText = $('saveText');
  const resetBtn = $('resetBtn');

  const zoomInBtn = $('zoomIn');
  const zoomOutBtn = $('zoomOut');
  const zoomLevel = $('zoomLevel');
  const fitWidthBtn = $('fitWidth');

  const editProps = $('editProps');
  const propsLabel = $('propsLabel');
  const toolHint = $('toolHint');
  const edFont = $('edFont');
  const edSize = $('edSize');
  const edSizeUp = $('edSizeUp');
  const edSizeDown = $('edSizeDown');
  const edBold = $('edBold');
  const edItalic = $('edItalic');
  const edColor = $('edColor');
  const edColorDot = $('edColorDot');
  const edAlignBtns = Array.from(document.querySelectorAll('.ed-align .ed-toggle'));

  const rpFileName = $('rpFileName');
  const rpFileSub = $('rpFileSub');
  const rpDirty = $('rpDirty');
  const rpResult = $('rpResult');

  const stPage = $('stPage');
  const stTotal = $('stTotal');
  const stZoom = $('stZoom');
  const stEdits = $('stEdits');
  const stStatus = $('stStatus');
  const stDot = $('stDot');

  // ---- State ---------------------------------------------------------------
  let state = null;     // { jobId, fileName, pages:[], spans:[], edits:{}, objects:[] }
  let zoom = 1;
  let fitMode = 'width'; // 'width' | null
  let tool = 'select';
  let selectedObj = null;
  let pendingImage = null;
  let objSeq = 1;

  const defaults = { font: '', size: 14, bold: false, italic: false, color: 0x111111, align: 0 };

  const toolHints = {
    select: 'Drag objects to move, grab a corner to resize, or press Delete to remove. Click highlighted text to edit it.',
    edit_text: 'Click any highlighted text run to replace it in place with the original font.',
    add_text: 'Drag a box anywhere on the page (or just click) to add a new text box, then type.',
    add_image: 'Pick an image, then drag a box on the page to place it.',
    add_link: 'Drag a box to create a clickable link, type the label and paste a URL.',
    erase: 'Drag over anything you want to white-out / cover up.',
  };

  // ---- Utilities -----------------------------------------------------------
  function escapeHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, (c) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }
  function isPdf(file) {
    return !!file && (file.type === 'application/pdf' || /\.pdf$/i.test(file.name || ''));
  }
  function intToHex(n) { n = (n || 0) & 0xFFFFFF; return '#' + n.toString(16).padStart(6, '0'); }
  function hexToInt(h) { return parseInt((h || '#000000').replace('#', ''), 16) & 0xFFFFFF; }

  function toast(msg, ok) {
    toastEl.innerHTML = (ok ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>' : '') + escapeHtml(msg);
    toastEl.classList.add('show');
    clearTimeout(toast._t);
    toast._t = setTimeout(() => toastEl.classList.remove('show'), 2400);
  }

  function setStatus(msg, kind) {
    stStatus.innerHTML = msg;
    stDot.className = 'dot' + (kind ? ' ' + kind : '');
  }
  function setDzStatus(msg, kind) {
    dzStatus.innerHTML = msg;
    dzStatus.className = 'dz-status' + (kind ? ' ' + kind : '');
  }

  function dlIcon() {
    return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>';
  }

  // ---------------------------------------------------------------------------
  //  Opening a PDF (upload or by job id)
  // ---------------------------------------------------------------------------
  function pickPdf(onPick) {
    fileInput.value = '';
    fileInput.onchange = () => {
      const f = fileInput.files && fileInput.files[0];
      if (f) onPick(f);
    };
    fileInput.click();
  }

  async function openFile(file) {
    if (!isPdf(file)) { setDzStatus('Please choose a PDF file.', 'error'); return; }
    showStart(true);
    setDzStatus('<span class="spinner"></span> Opening PDF…', 'busy');
    setStatus('<span class="spinner"></span> Opening PDF…', 'busy');
    try {
      const fd = new FormData();
      fd.append('file', file, file.name);
      const res = await fetch(API + '/open', { method: 'POST', body: fd });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || "Couldn't open that PDF.");
      loadState(data, file.name);
    } catch (err) {
      setDzStatus(escapeHtml(err.message || 'Something went wrong.'), 'error');
      setStatus(escapeHtml(err.message || 'Something went wrong.'), 'error');
    }
  }

  // Re-open an already-staged job (passed via ?job=<id> from the inline tool).
  async function openByJob(jobId, fileName) {
    showStart(true);
    setDzStatus('<span class="spinner"></span> Loading your document…', 'busy');
    setStatus('<span class="spinner"></span> Loading…', 'busy');
    try {
      const res = await fetch(API + '/' + encodeURIComponent(jobId) + '/state');
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || "Couldn't load that document.");
      loadState(data, fileName || 'document.pdf');
    } catch (err) {
      // Fall back to the open screen so the user can pick a file.
      setDzStatus('That editing session expired — choose a PDF to continue.', 'error');
      setStatus('Open a PDF to start editing.', '');
    }
  }

  function loadState(data, fileName) {
    state = {
      jobId: data.job_id,
      fileName: fileName || 'document.pdf',
      pages: data.pages || [],
      spans: data.spans || [],
      edits: {},
      objects: [],
    };
    selectedObj = null;
    pendingImage = null;
    objSeq = 1;
    rpResult.classList.add('hidden');

    rpFileName.textContent = state.fileName;
    rpFileName.title = state.fileName;

    if (!data.has_text || !state.spans.length) {
      rpFileSub.textContent = 'No selectable text — add text/images or Run OCR to edit existing text.';
      setStatus('Scanned PDF (no selectable text). Add objects or Run OCR.', 'warn');
    } else {
      rpFileSub.textContent = state.spans.length + ' editable text run' + (state.spans.length === 1 ? '' : 's') + ' · ' + state.pages.length + ' page' + (state.pages.length === 1 ? '' : 's');
      setStatus('Loaded. Pick a tool and start editing.', 'ok');
    }

    showStart(false);
    setTool('select');
    fitMode = 'width';
    buildPages();
    buildThumbs();
    updateDirty();
    stTotal.textContent = state.pages.length;
    stPage.textContent = state.pages.length ? 1 : '–';
  }

  function showStart(show) {
    startScreen.classList.toggle('hidden', !show);
  }

  // ---------------------------------------------------------------------------
  //  Page rendering
  // ---------------------------------------------------------------------------
  function buildPages() {
    canvasScroll.innerHTML = '';
    if (!state || !state.pages.length) return;
    state.pages.forEach((page) => {
      const wrap = document.createElement('div');
      wrap.className = 'ed-page';
      wrap.dataset.page = page.page;

      const num = document.createElement('span');
      num.className = 'ed-pnum';
      num.textContent = 'Page ' + page.page;
      wrap.appendChild(num);

      const img = document.createElement('img');
      img.alt = 'Page ' + page.page;
      img.loading = 'lazy';
      img.src = page.preview_url;
      wrap.appendChild(img);

      const overlay = document.createElement('div');
      overlay.className = 'edit-overlay';
      overlay.dataset.page = page.page;
      wireOverlayDrawing(overlay, page);
      wrap.appendChild(overlay);

      canvasScroll.appendChild(wrap);
    });
    applyZoom();
  }

  function buildThumbs() {
    thumbs.innerHTML = '';
    if (!state || !state.pages.length) { $('leftCount').textContent = 'No document'; return; }
    state.pages.forEach((page) => {
      const el = document.createElement('div');
      el.className = 'thumb' + (page.page === 1 ? ' active' : '');
      el.dataset.page = page.page;
      el.innerHTML = '<div class="thumb-prev"><img loading="lazy" alt="Page ' + page.page + '" src="' + page.preview_url + '"></div>' +
        '<div class="thumb-num">Page ' + page.page + '</div>';
      el.addEventListener('click', () => scrollToPage(page.page));
      thumbs.appendChild(el);
    });
    $('leftCount').textContent = state.pages.length + ' page' + (state.pages.length === 1 ? '' : 's');
  }

  function scrollToPage(n) {
    const wrap = canvasScroll.querySelector('.ed-page[data-page="' + n + '"]');
    if (wrap) wrap.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function computeFitWidthZoom() {
    if (!state || !state.pages.length) return 1;
    const avail = Math.max(200, canvasScroll.clientWidth - 48);
    const widest = Math.max.apply(null, state.pages.map((p) => p.width));
    return avail / widest;
  }

  function applyZoom() {
    if (!state) return;
    if (fitMode === 'width') zoom = computeFitWidthZoom();
    zoom = Math.max(0.2, Math.min(5, zoom));
    state.pages.forEach((page) => {
      const wrap = canvasScroll.querySelector('.ed-page[data-page="' + page.page + '"]');
      if (!wrap) return;
      wrap.style.width = (page.width * zoom) + 'px';
      wrap.style.height = (page.height * zoom) + 'px';
      const overlay = wrap.querySelector('.edit-overlay');
      layoutSpans(page, overlay);
      renderObjects(page, overlay);
    });
    const pct = Math.round(zoom * 100) + '%';
    zoomLevel.textContent = pct;
    stZoom.textContent = pct;
    fitWidthBtn.classList.toggle('active', fitMode === 'width');
  }

  // ---------------------------------------------------------------------------
  //  Existing text runs (edit_text)
  // ---------------------------------------------------------------------------
  function layoutSpans(page, overlay) {
    if (!overlay) return;
    overlay.querySelectorAll('.edit-span, .edit-span-input').forEach((n) => n.remove());
    const scale = zoom;
    state.spans.filter((s) => s.page === page.page).forEach((s) => {
      const [x0, y0, x1, y1] = s.bbox;
      const div = document.createElement('div');
      div.className = 'edit-span' + (state.edits[s.id] !== undefined ? ' changed' : '');
      div.style.left = (x0 * scale) + 'px';
      div.style.top = (y0 * scale) + 'px';
      div.style.width = Math.max(6, (x1 - x0) * scale) + 'px';
      div.style.height = Math.max(6, (y1 - y0) * scale) + 'px';
      div.title = 'Click to edit text';
      div.addEventListener('click', (e) => { e.stopPropagation(); beginEditSpan(s, div, overlay, scale); });
      overlay.appendChild(div);
    });
  }

  function beginEditSpan(span, div, overlay, scale) {
    canvasScroll.querySelectorAll('.edit-span-input').forEach((i) => i.remove());
    canvasScroll.querySelectorAll('.edit-span.editing').forEach((d) => d.classList.remove('editing'));
    div.classList.add('editing');

    const [x0, y0, x1, y1] = span.bbox;
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'edit-span-input';
    input.value = state.edits[span.id] !== undefined ? state.edits[span.id] : span.text;
    input.style.left = (x0 * scale) + 'px';
    input.style.top = (y0 * scale) + 'px';
    input.style.width = Math.max(40, (x1 - x0) * scale + 20) + 'px';
    input.style.height = Math.max(18, (y1 - y0) * scale + 4) + 'px';
    input.style.fontSize = Math.max(9, (y1 - y0) * scale * 0.8) + 'px';
    overlay.appendChild(input);
    input.focus();
    input.select();

    const commit = () => {
      const val = input.value;
      if (val !== span.text) state.edits[span.id] = val;
      else delete state.edits[span.id];
      input.remove();
      div.classList.remove('editing');
      div.classList.toggle('changed', state.edits[span.id] !== undefined);
      updateDirty();
    };
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); commit(); }
      else if (e.key === 'Escape') { input.remove(); div.classList.remove('editing'); }
    });
    input.addEventListener('blur', commit);
  }

  // ---------------------------------------------------------------------------
  //  Added objects (add_text / add_image / add_link / erase)
  // ---------------------------------------------------------------------------
  function renderObjects(page, overlay) {
    if (!overlay) return;
    overlay.querySelectorAll('.ed-obj').forEach((n) => n.remove());
    const scale = zoom;
    state.objects.filter((o) => o.page === page.page).forEach((o) => overlay.appendChild(buildObjectEl(o, scale)));
  }

  function positionObjEl(el, o, scale) {
    const [x0, y0, x1, y1] = o.bbox;
    el.style.left = (x0 * scale) + 'px';
    el.style.top = (y0 * scale) + 'px';
    el.style.width = Math.max(8, (x1 - x0) * scale) + 'px';
    el.style.height = Math.max(8, (y1 - y0) * scale) + 'px';
    const body = el.querySelector('.ed-obj-body');
    if (body && (o.type === 'add_text' || o.type === 'add_link')) {
      body.style.fontSize = Math.max(6, (o.size || 12) * scale) + 'px';
      body.style.fontWeight = o.bold ? '700' : '400';
      body.style.fontStyle = o.italic ? 'italic' : 'normal';
      body.style.textAlign = ['left', 'center', 'right'][o.align || 0];
      if (o.type === 'add_text') body.style.color = intToHex(o.color);
    }
  }

  function buildObjectEl(o, scale) {
    const el = document.createElement('div');
    el.className = 'ed-obj' + (selectedObj === o ? ' selected' : '');
    el.dataset.type = o.type;
    el.dataset.id = o.id;
    positionObjEl(el, o, scale);

    const move = document.createElement('div');
    move.className = 'ed-obj-move';
    move.innerHTML = '<span>move</span><button type="button" class="od-del" title="Delete">✕</button>';
    el.appendChild(move);

    const body = document.createElement('div');
    body.className = 'ed-obj-body';
    if (o.type === 'add_image') {
      const im = document.createElement('img');
      im.src = o.image_b64;
      body.appendChild(im);
    } else if (o.type === 'erase') {
      // plain white box
    } else {
      body.contentEditable = 'true';
      body.spellcheck = false;
      body.textContent = o.text || '';
      if (!o.text) { body.classList.add('ed-obj-empty'); body.dataset.placeholder = o.type === 'add_link' ? 'Link text' : 'Type here'; }
      body.addEventListener('input', () => {
        o.text = body.innerText.replace(/\n$/, '');
        body.classList.toggle('ed-obj-empty', !o.text);
        updateDirty();
      });
      body.addEventListener('focus', () => selectObject(o));
      body.addEventListener('pointerdown', (e) => e.stopPropagation());
    }
    el.appendChild(body);

    if (o.type === 'add_link') {
      const lb = document.createElement('div');
      lb.className = 'ed-linkbar';
      lb.innerHTML = '<label>URL</label>';
      const urlInput = document.createElement('input');
      urlInput.type = 'url';
      urlInput.placeholder = 'https://example.com';
      urlInput.value = o.url || '';
      urlInput.addEventListener('input', () => { o.url = urlInput.value.trim(); updateDirty(); });
      urlInput.addEventListener('pointerdown', (e) => e.stopPropagation());
      lb.appendChild(urlInput);
      el.appendChild(lb);
    }

    const handle = document.createElement('div');
    handle.className = 'ed-obj-handle';
    el.appendChild(handle);

    el.addEventListener('pointerdown', (e) => {
      if (e.target.closest('.od-del') || e.target.closest('.ed-obj-handle') || e.target.closest('.ed-obj-move')) return;
      selectObject(o);
    });
    move.querySelector('.od-del').addEventListener('click', (e) => { e.stopPropagation(); deleteObject(o); });
    move.addEventListener('pointerdown', (e) => startDragObject(e, o, el, 'move'));
    handle.addEventListener('pointerdown', (e) => startDragObject(e, o, el, 'resize'));

    return el;
  }

  function selectObject(o) {
    selectedObj = o;
    canvasScroll.querySelectorAll('.ed-obj').forEach((el) => el.classList.toggle('selected', el.dataset.id === o.id));
    syncPropsFromObject(o);
    updatePropsVisibility();
  }
  function deselectAll() {
    selectedObj = null;
    canvasScroll.querySelectorAll('.ed-obj.selected').forEach((el) => el.classList.remove('selected'));
    updatePropsVisibility();
  }
  function deleteObject(o) {
    state.objects = state.objects.filter((x) => x !== o);
    if (selectedObj === o) selectedObj = null;
    applyZoom();
    updateDirty();
  }

  function startDragObject(e, o, el, mode) {
    e.preventDefault();
    e.stopPropagation();
    selectObject(o);
    const scale = zoom;
    const startX = e.clientX, startY = e.clientY;
    const orig = o.bbox.slice();
    const page = state.pages.find((p) => p.page === o.page);
    const move = (ev) => {
      const dx = (ev.clientX - startX) / scale;
      const dy = (ev.clientY - startY) / scale;
      if (mode === 'move') {
        let nx0 = orig[0] + dx, ny0 = orig[1] + dy;
        const w = orig[2] - orig[0], h = orig[3] - orig[1];
        nx0 = Math.max(0, Math.min(page.width - w, nx0));
        ny0 = Math.max(0, Math.min(page.height - h, ny0));
        o.bbox = [nx0, ny0, nx0 + w, ny0 + h];
      } else {
        const nx1 = Math.max(orig[0] + 8, Math.min(page.width, orig[2] + dx));
        const ny1 = Math.max(orig[1] + 8, Math.min(page.height, orig[3] + dy));
        o.bbox = [orig[0], orig[1], nx1, ny1];
      }
      positionObjEl(el, o, scale);
    };
    const up = () => {
      document.removeEventListener('pointermove', move);
      document.removeEventListener('pointerup', up);
      updateDirty();
    };
    document.addEventListener('pointermove', move);
    document.addEventListener('pointerup', up);
  }

  // Marquee drawing on an overlay for the create-tools.
  function wireOverlayDrawing(overlay, page) {
    overlay.addEventListener('pointerdown', (e) => {
      if (tool === 'select' || tool === 'edit_text') {
        if (!e.target.closest('.ed-obj') && !e.target.closest('.edit-span')) deselectAll();
        return;
      }
      if (e.target.closest('.ed-obj')) return;
      e.preventDefault();
      const scale = zoom;
      const rect = overlay.getBoundingClientRect();
      const sx = e.clientX - rect.left, sy = e.clientY - rect.top;

      const marquee = document.createElement('div');
      marquee.className = 'ed-marquee';
      overlay.appendChild(marquee);
      let cx = sx, cy = sy;

      const move = (ev) => {
        cx = ev.clientX - rect.left; cy = ev.clientY - rect.top;
        const x = Math.min(sx, cx), y = Math.min(sy, cy);
        marquee.style.left = x + 'px';
        marquee.style.top = y + 'px';
        marquee.style.width = Math.abs(cx - sx) + 'px';
        marquee.style.height = Math.abs(cy - sy) + 'px';
      };
      const up = () => {
        document.removeEventListener('pointermove', move);
        document.removeEventListener('pointerup', up);
        marquee.remove();
        let x = Math.min(sx, cx), y = Math.min(sy, cy);
        let w = Math.abs(cx - sx), h = Math.abs(cy - sy);
        if (w < 6 || h < 6) {
          if (tool === 'add_text' || tool === 'add_link') { w = 180; h = 26; }
          else if (tool === 'erase') { w = 120; h = 24; }
          else { return; } // image needs a real box
        }
        const bbox = [x / scale, y / scale, (x + w) / scale, (y + h) / scale];
        createObject(tool, page.page, bbox);
      };
      document.addEventListener('pointermove', move);
      document.addEventListener('pointerup', up);
    });
  }

  function createObject(type, page, bbox) {
    if (type === 'add_image' && !pendingImage) {
      setStatus('Pick an image first, then drag a box to place it.', 'warn');
      return;
    }
    const o = {
      id: 'obj' + (objSeq++),
      type, page, bbox,
      text: '',
      font: defaults.font,
      size: defaults.size,
      color: defaults.color,
      bold: defaults.bold,
      italic: defaults.italic,
      align: defaults.align,
      url: '',
      image_b64: type === 'add_image' ? pendingImage : null,
    };
    state.objects.push(o);
    if (type === 'add_image') { pendingImage = null; setTool('select'); }
    applyZoom();
    selectObject(o);
    if (type === 'add_text' || type === 'add_link') {
      const el = canvasScroll.querySelector('.ed-obj[data-id="' + o.id + '"] .ed-obj-body');
      if (el) el.focus();
    }
    updateDirty();
  }

  // ---------------------------------------------------------------------------
  //  Tool selection
  // ---------------------------------------------------------------------------
  function setTool(t) {
    tool = t;
    document.querySelectorAll('.tb-btn[data-tool]').forEach((b) => b.classList.toggle('active', b.dataset.tool === t));
    document.querySelectorAll('.ed-tool[data-tool]').forEach((b) => b.classList.toggle('active', b.dataset.tool === t));
    ['select', 'edit_text', 'add_text', 'add_image', 'add_link', 'erase'].forEach((x) =>
      canvasScroll.classList.toggle('tool-' + x, x === t));
    toolHint.textContent = toolHints[t] || toolHints.select;
    updatePropsVisibility();
    if (t === 'add_image') {
      imgInput.value = '';
      imgInput.click();
    }
  }

  function updatePropsVisibility() {
    const textTool = tool === 'add_text' || tool === 'add_link';
    const textSel = selectedObj && (selectedObj.type === 'add_text' || selectedObj.type === 'add_link');
    const show = (textTool || textSel) ? '' : 'none';
    editProps.style.display = show;
    propsLabel.style.display = show;
  }

  // ---------------------------------------------------------------------------
  //  Property bar
  // ---------------------------------------------------------------------------
  function syncPropsFromObject(o) {
    if (!o || (o.type !== 'add_text' && o.type !== 'add_link')) return;
    edFont.value = o.font || '';
    edSize.value = Math.round(o.size || 12);
    edBold.classList.toggle('active', !!o.bold);
    edItalic.classList.toggle('active', !!o.italic);
    edColor.value = intToHex(o.color);
    edColorDot.style.background = intToHex(o.color);
    edAlignBtns.forEach((b) => b.classList.toggle('active', String(o.align || 0) === b.dataset.align));
  }
  function applyProp(fn) {
    if (selectedObj && (selectedObj.type === 'add_text' || selectedObj.type === 'add_link')) {
      fn(selectedObj);
      const el = canvasScroll.querySelector('.ed-obj[data-id="' + selectedObj.id + '"]');
      if (el) positionObjEl(el, selectedObj, zoom);
      updateDirty();
    }
  }
  function setSize(v) {
    v = Math.max(4, Math.min(200, v));
    edSize.value = v;
    defaults.size = v;
    applyProp((o) => o.size = v);
  }

  // ---------------------------------------------------------------------------
  //  Save (apply) / reset
  // ---------------------------------------------------------------------------
  function buildOperations() {
    const ops = [];
    Object.keys(state.edits).forEach((id) => {
      const span = state.spans.find((s) => s.id === id);
      if (!span) return;
      ops.push({ type: 'edit_text', page: span.page, bbox: span.bbox, text: state.edits[id], font: span.font, size: span.size, color: span.color });
    });
    state.objects.forEach((o) => {
      const base = { type: o.type, page: o.page, bbox: o.bbox };
      if (o.type === 'add_text') Object.assign(base, { text: o.text || '', font: o.font || '', size: o.size, color: o.color, bold: o.bold, italic: o.italic, align: o.align });
      else if (o.type === 'add_link') Object.assign(base, { text: o.text || '', url: o.url || '', size: o.size, color: o.color, align: o.align });
      else if (o.type === 'add_image') Object.assign(base, { image_b64: o.image_b64 });
      else if (o.type === 'erase') Object.assign(base, { fill: 0xFFFFFF });
      if (o.type === 'add_text' && !o.text) return;
      if (o.type === 'add_link' && !o.text && !o.url) return;
      if (o.type === 'add_image' && !o.image_b64) return;
      ops.push(base);
    });
    return ops;
  }

  function editCount() {
    if (!state) return 0;
    return Object.keys(state.edits).length + state.objects.length;
  }

  function updateDirty() {
    const n = editCount();
    saveBtn.disabled = n === 0;
    saveText.textContent = n > 0 ? 'Save & Download (' + n + ')' : 'Save & Download';
    rpDirty.classList.toggle('hidden', n === 0);
    rpDirty.textContent = '● ' + n + ' unsaved change' + (n === 1 ? '' : 's');
    stEdits.textContent = n;
  }

  async function save() {
    if (!state) return;
    const operations = buildOperations();
    if (!operations.length) { setStatus('Nothing to save yet — make an edit first.', 'warn'); return; }
    saveBtn.disabled = true;
    rpResult.classList.add('hidden');
    setStatus('<span class="spinner"></span> Saving your edits with the original fonts…', 'busy');
    try {
      const res = await fetch(API + '/apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: state.jobId, operations }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || "Couldn't apply edits.");
      rpResult.innerHTML =
        '<div>Edits applied: <b>' + data.edits_applied + '</b></div>' +
        '<a class="dl-link" href="' + data.download_url + '" download>' + dlIcon() + ' Download edited PDF</a>';
      rpResult.classList.remove('hidden');
      setStatus('Saved. Download your edited PDF from the panel.', 'ok');
      toast('Saved ' + data.edits_applied + ' edit' + (data.edits_applied === 1 ? '' : 's'), true);
      // Auto-trigger the download for convenience.
      const a = document.createElement('a');
      a.href = data.download_url; a.download = '';
      document.body.appendChild(a); a.click(); a.remove();
    } catch (err) {
      setStatus(escapeHtml(err.message || 'Something went wrong.'), 'error');
    } finally {
      saveBtn.disabled = editCount() === 0;
    }
  }

  // ---------------------------------------------------------------------------
  //  OCR
  // ---------------------------------------------------------------------------
  async function runOcr(file) {
    if (!isPdf(file)) { setStatus('Please choose a PDF file.', 'error'); return; }
    showStart(true);
    setDzStatus('<span class="spinner"></span> Running OCR (this can take a moment on big scans)…', 'busy');
    setStatus('<span class="spinner"></span> Running OCR…', 'busy');
    try {
      const fd = new FormData();
      fd.append('file', file, file.name);
      const res = await fetch(API + '/ocr', { method: 'POST', body: fd });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || 'OCR failed.');
      toast(data.note || 'OCR complete', true);
      // The OCR'd file becomes the editable source for its job; reopen it.
      await openByJob(data.job_id, (file.name || 'document').replace(/\.pdf$/i, '') + '_ocr.pdf');
    } catch (err) {
      setDzStatus(escapeHtml(err.message || 'Something went wrong.'), 'error');
      setStatus(escapeHtml(err.message || 'Something went wrong.'), 'error');
    }
  }

  // ---------------------------------------------------------------------------
  //  Wiring
  // ---------------------------------------------------------------------------
  function wireDropzone() {
    const browse = () => pickPdf(openFile);
    dropzone.addEventListener('click', (e) => { if (!e.target.closest('button')) browse(); });
    dropzone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); browse(); } });
    $('dzOpen').addEventListener('click', (e) => { e.stopPropagation(); browse(); });
    $('dzOcr').addEventListener('click', (e) => { e.stopPropagation(); pickPdf(runOcr); });

    ['dragenter', 'dragover'].forEach((ev) =>
      dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add('drag'); }));
    ['dragleave', 'drop'].forEach((ev) =>
      dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove('drag'); }));
    dropzone.addEventListener('drop', (e) => {
      const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) openFile(f);
    });
  }

  function wireToolbar() {
    document.querySelectorAll('.tb-btn[data-tool], .ed-tool[data-tool]').forEach((b) =>
      b.addEventListener('click', () => setTool(b.dataset.tool)));

    openBtn.addEventListener('click', () => pickPdf(openFile));
    ocrBtn.addEventListener('click', () => pickPdf(runOcr));
    saveBtn.addEventListener('click', save);
    resetBtn.addEventListener('click', () => {
      if (!state) return;
      state.edits = {};
      state.objects = [];
      selectedObj = null;
      applyZoom();
      updateDirty();
      rpResult.classList.add('hidden');
      setStatus('All edits cleared.', '');
    });

    zoomInBtn.addEventListener('click', () => { fitMode = null; zoom *= 1.2; applyZoom(); });
    zoomOutBtn.addEventListener('click', () => { fitMode = null; zoom /= 1.2; applyZoom(); });
    fitWidthBtn.addEventListener('click', () => { fitMode = 'width'; applyZoom(); });

    imgInput.addEventListener('change', () => {
      const f = imgInput.files && imgInput.files[0];
      if (!f) { setTool('select'); return; }
      const reader = new FileReader();
      reader.onload = () => {
        pendingImage = reader.result;
        setStatus('Image ready — now drag a box on the page to place it.', 'ok');
      };
      reader.readAsDataURL(f);
    });
  }

  function wireProps() {
    edFont.addEventListener('change', () => { defaults.font = edFont.value; applyProp((o) => o.font = edFont.value); });
    edSize.addEventListener('change', () => setSize(parseInt(edSize.value, 10) || 14));
    edSizeUp.addEventListener('click', () => setSize((parseInt(edSize.value, 10) || 14) + 1));
    edSizeDown.addEventListener('click', () => setSize((parseInt(edSize.value, 10) || 14) - 1));
    edBold.addEventListener('click', () => { const on = !edBold.classList.contains('active'); edBold.classList.toggle('active', on); defaults.bold = on; applyProp((o) => o.bold = on); });
    edItalic.addEventListener('click', () => { const on = !edItalic.classList.contains('active'); edItalic.classList.toggle('active', on); defaults.italic = on; applyProp((o) => o.italic = on); });
    edColor.addEventListener('input', () => { const v = hexToInt(edColor.value); edColorDot.style.background = edColor.value; defaults.color = v; applyProp((o) => o.color = v); });
    edAlignBtns.forEach((b) => b.addEventListener('click', () => {
      edAlignBtns.forEach((x) => x.classList.remove('active'));
      b.classList.add('active');
      const a = parseInt(b.dataset.align, 10) || 0;
      defaults.align = a;
      applyProp((o) => o.align = a);
    }));
  }

  function wireScrollSync() {
    let t = null;
    canvasScroll.addEventListener('scroll', () => {
      clearTimeout(t);
      t = setTimeout(() => {
        const mid = canvasScroll.getBoundingClientRect().top + canvasScroll.clientHeight / 2;
        let best = null, bestDist = Infinity;
        canvasScroll.querySelectorAll('.ed-page').forEach((c) => {
          const r = c.getBoundingClientRect();
          const center = r.top + r.height / 2;
          const d = Math.abs(center - mid);
          if (d < bestDist) { bestDist = d; best = c; }
        });
        if (best) {
          const n = best.dataset.page;
          stPage.textContent = n;
          thumbs.querySelectorAll('.thumb').forEach((th) => th.classList.toggle('active', th.dataset.page === n));
        }
      }, 80);
    });
  }

  function wireKeyboard() {
    document.addEventListener('keydown', (e) => {
      const meta = e.ctrlKey || e.metaKey;
      const typing = document.activeElement && (document.activeElement.isContentEditable ||
        ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName));

      if (meta && e.key.toLowerCase() === 's') { e.preventDefault(); save(); return; }
      if (meta && (e.key === '=' || e.key === '+')) { e.preventDefault(); fitMode = null; zoom *= 1.2; applyZoom(); return; }
      if (meta && e.key === '-') { e.preventDefault(); fitMode = null; zoom /= 1.2; applyZoom(); return; }

      if (typing) return;

      if (e.key === 'Escape') { deselectAll(); return; }
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedObj) { e.preventDefault(); deleteObject(selectedObj); return; }

      const map = { v: 'select', t: 'edit_text' };
      if (!meta && map[e.key.toLowerCase()]) { setTool(map[e.key.toLowerCase()]); }
    });

    window.addEventListener('beforeunload', (e) => {
      if (editCount() > 0) { e.preventDefault(); e.returnValue = ''; return ''; }
    });

    let rt = null;
    window.addEventListener('resize', () => {
      if (!state) return;
      clearTimeout(rt);
      rt = setTimeout(() => { if (fitMode === 'width') applyZoom(); }, 120);
    });
  }

  // ---------------------------------------------------------------------------
  //  Theme (shared with the main app via the 'mcq-theme' localStorage key)
  // ---------------------------------------------------------------------------
  function wireTheme() {
    const KEY = 'mcq-theme';
    const VALID = ['system', 'light', 'dark'];
    const root = document.documentElement;
    const sw = $('themeSwitch');
    const meta = $('themeColorMeta');
    const mql = window.matchMedia('(prefers-color-scheme: light)');

    function stored() {
      let t;
      try { t = localStorage.getItem(KEY); } catch (e) { t = null; }
      return VALID.includes(t) ? t : 'system';
    }
    function effective(choice) {
      if (choice === 'system') return mql.matches ? 'light' : 'dark';
      return choice;
    }
    function syncButtons(choice) {
      if (!sw) return;
      sw.querySelectorAll('button[data-theme-value]').forEach((b) =>
        b.setAttribute('aria-pressed', String(b.getAttribute('data-theme-value') === choice)));
    }
    function syncMeta(choice) {
      if (meta) meta.setAttribute('content', effective(choice) === 'light' ? '#ffffff' : '#1A1A2E');
    }
    function apply(choice, persist) {
      const t = VALID.includes(choice) ? choice : 'system';
      root.setAttribute('data-theme', t);
      if (persist) { try { localStorage.setItem(KEY, t); } catch (e) {} }
      syncButtons(t);
      syncMeta(t);
    }
    if (sw) {
      sw.addEventListener('click', (e) => {
        const b = e.target.closest('button[data-theme-value]');
        if (b) apply(b.getAttribute('data-theme-value'), true);
      });
    }
    const onScheme = () => { if (stored() === 'system') syncMeta('system'); };
    if (mql.addEventListener) mql.addEventListener('change', onScheme);
    else if (mql.addListener) mql.addListener(onScheme);

    apply(stored(), false);
  }

  // ---------------------------------------------------------------------------
  //  Init
  // ---------------------------------------------------------------------------
  function init() {
    wireTheme();
    wireDropzone();
    wireToolbar();
    wireProps();
    wireScrollSync();
    wireKeyboard();
    updatePropsVisibility();
    edColorDot.style.background = intToHex(defaults.color);

    // If we were opened from the inline tool with ?job=<id>, load that PDF.
    const params = new URLSearchParams(location.search);
    const job = params.get('job');
    const name = params.get('name');
    if (job) {
      openByJob(job, name);
    } else {
      showStart(true);
      setStatus('Open a PDF to start editing.', '');
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
