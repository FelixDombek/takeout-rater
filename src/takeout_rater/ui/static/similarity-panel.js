/**
 * SimilarityPanel – "More like this" panel logic.
 *
 * Usage:
 *   SimilarityPanel.init({ assetId: 42 });
 *
 * Once initialised, the panel reads DOM elements by the IDs defined in
 * the accompanying detail.html template.  Two operating modes are supported:
 *
 *   1. Asset mode   (default) – calls GET  /api/assets/{id}/similar
 *   2. Reverse mode (loaded)  – calls POST /api/reverse-search with the file
 *
 * External API (exposed on the returned object):
 *   panel.setReferenceFile(file | null)  – switch to/from reverse mode
 */
var SimilarityPanel = (function () {
  'use strict';

  function init(opts) {
    var assetId = opts.assetId;

    var panelEl = document.getElementById('mlt-panel');
    if (!panelEl) return;

    var grid         = document.getElementById('mlt-grid');
    var statusEl     = document.getElementById('mlt-status');
    var headingEl    = document.getElementById('mlt-heading');

    var currentSort    = 'score';
    var currentResults = [];
    var debounceTimer  = null;
    var referenceFile  = null;   // non-null → reverse-search mode

    // ── Per-metric defaults (mirrors Python backend defaults) ──────────────
    var _clipMetricDefaults = {
      cosine:    { label: 'Min cosine similarity:', hint: 'range 0–1, default 0.85',       val: '0.85', min: '0',    max: '1',    step: '0.01' },
      euclidean: { label: 'Max L2 distance:',       hint: 'range 0–2, default 0.45',       val: '0.45', min: '0',    max: '2',    step: '0.01' },
      combined:  { label: 'Max angle (radians):',   hint: 'range 0–π≈3.14, default 0.46', val: '0.46', min: '0',    max: '3.15', step: '0.01' }
    };

    // ── Helper: get current method from tab buttons ────────────────────────
    function getMethod() {
      var active = panelEl.querySelector('.mlt-tab-btn[data-role="method"].active');
      return active ? active.dataset.value : 'clip';
    }

    // ── Control change handlers ─────────────────────────────────────────────

    function onMethodChange(btn) {
      panelEl.querySelectorAll('.mlt-tab-btn[data-role="method"]').forEach(function (b) {
        b.classList.toggle('active', b === btn);
      });
      var method = btn.dataset.value;
      var phashOpts = document.getElementById('mlt-phash-opts');
      var clipOpts  = document.getElementById('mlt-clip-opts');
      var clipThRow = document.getElementById('mlt-clip-threshold-row');
      if (phashOpts) phashOpts.style.display = method === 'phash' ? '' : 'none';
      if (clipOpts)  clipOpts.style.display  = method === 'clip'  ? '' : 'none';
      if (clipThRow) clipThRow.style.display = method === 'clip'  ? '' : 'none';
      scheduleSimilar();
    }

    function onMetricChange() {
      var metric = document.getElementById('mlt-clip-metric').value;
      var cfg = _clipMetricDefaults[metric] || _clipMetricDefaults.cosine;
      var labelEl = document.getElementById('mlt-clip-threshold-label');
      var hintEl  = document.getElementById('mlt-clip-threshold-hint');
      var inp     = document.getElementById('mlt-clip-threshold');
      if (labelEl) labelEl.textContent = cfg.label;
      if (hintEl)  hintEl.textContent  = cfg.hint;
      if (inp) {
        inp.value = cfg.val;
        inp.min   = cfg.min;
        inp.max   = cfg.max;
        inp.step  = cfg.step;
      }
      scheduleSimilar();
    }

    function setSort(sortKey) {
      currentSort = sortKey;
      panelEl.querySelectorAll('.mlt-tab-btn[data-role="sort"]').forEach(function (b) {
        b.classList.toggle('active', b.dataset.value === sortKey);
      });
      renderResults();
    }

    // ── Expose to inline onclick attributes ───────────────────────────────
    window.mltOnMethodTab   = onMethodChange;
    window.mltOnMetricChange = onMetricChange;
    window.mltSetSort       = setSort;

    // Reload on threshold input (debounced)
    ['mlt-phash-threshold', 'mlt-clip-threshold'].forEach(function (id) {
      var el = document.getElementById(id);
      if (el) el.addEventListener('input', function () { scheduleSimilar(); });
    });

    // ── Score badge helpers ────────────────────────────────────────────────

    function scoreBadgeText(item, method, metric) {
      if (method === 'phash') return item.score + ' bits';
      if (metric === 'cosine') return 'sim=' + item.score.toFixed(3);
      if (metric === 'euclidean') return 'L2=' + item.score.toFixed(3);
      return item.score.toFixed(3) + ' rad';
    }

    function scoreBadgeTitle(method, metric) {
      if (method === 'phash') return 'Hamming distance (bits)';
      if (metric === 'cosine') return 'Cosine similarity';
      if (metric === 'euclidean') return 'L2 (Euclidean) distance';
      return 'Angular distance (radians)';
    }

    // ── Rendering ──────────────────────────────────────────────────────────

    function formatDate(ts) {
      if (!ts) return '';
      var d = new Date(ts * 1000);
      return d.getFullYear() + '-'
        + String(d.getMonth() + 1).padStart(2, '0') + '-'
        + String(d.getDate()).padStart(2, '0');
    }

    function renderResults() {
      grid.innerHTML = '';
      if (!currentResults.length) return;

      var method = getMethod();
      var metricEl = document.getElementById('mlt-clip-metric');
      var metric = metricEl ? metricEl.value : 'cosine';

      var sorted = currentResults.slice();
      if (currentSort === 'time') {
        sorted.sort(function (a, b) {
          var ta = a.taken_at != null ? a.taken_at : 0;
          var tb = b.taken_at != null ? b.taken_at : 0;
          return tb - ta;
        });
      }

      sorted.forEach(function (item) {
        var card = document.createElement('a');
        card.className = 'mlt-card';
        card.href = '/assets/' + item.asset_id;

        var img = document.createElement('img');
        img.className = 'mlt-thumb';
        img.src = '/thumbs/' + item.asset_id;
        img.alt = item.filename || '';
        img.loading = 'lazy';

        var placeholder = document.createElement('div');
        placeholder.className = 'mlt-thumb-placeholder';
        placeholder.style.display = 'none';
        placeholder.textContent = '🖼';
        img.onerror = function () {
          img.style.display = 'none';
          placeholder.style.display = 'flex';
        };

        var body = document.createElement('div');
        body.className = 'mlt-body';

        var scoreBadge = document.createElement('span');
        scoreBadge.className = 'mlt-score';
        scoreBadge.title = scoreBadgeTitle(method, metric);
        scoreBadge.textContent = scoreBadgeText(item, method, metric);
        body.appendChild(scoreBadge);

        var dateStr = formatDate(item.taken_at);
        if (dateStr) {
          var dateSpan = document.createElement('span');
          dateSpan.title = item.filename || '';
          dateSpan.textContent = dateStr;
          body.appendChild(dateSpan);
        } else {
          body.appendChild(document.createTextNode(item.filename || ''));
        }

        card.appendChild(img);
        card.appendChild(placeholder);
        card.appendChild(body);
        grid.appendChild(card);
      });
    }

    // ── API fetch ─────────────────────────────────────────────────────────

    function loadSimilar() {
      var method = getMethod();
      var metricEl = document.getElementById('mlt-clip-metric');
      var metric = metricEl ? metricEl.value : 'cosine';
      var threshold;
      if (method === 'phash') {
        var phEl = document.getElementById('mlt-phash-threshold');
        threshold = phEl ? (parseInt(phEl.value, 10) || 20) : 20;
      } else {
        var clEl = document.getElementById('mlt-clip-threshold');
        threshold = clEl ? (parseFloat(clEl.value) || 0.85) : 0.85;
      }

      statusEl.textContent = 'Loading…';
      statusEl.className = 'mlt-status';
      grid.innerHTML = '';

      var qs = '?method=' + encodeURIComponent(method)
             + '&metric=' + encodeURIComponent(metric)
             + '&threshold=' + threshold;

      var fetchPromise;
      if (referenceFile) {
        // Reverse-search mode: POST the file
        var fd = new FormData();
        fd.append('file', referenceFile);
        fetchPromise = fetch('/api/reverse-search' + qs, { method: 'POST', body: fd });
      } else {
        // Normal mode: GET similar assets for the current asset
        fetchPromise = fetch('/api/assets/' + assetId + '/similar' + qs);
      }

      fetchPromise
        .then(function (r) { return r.json(); })
        .then(function (data) {
          if (data.error === 'no_embedding') {
            statusEl.textContent = 'No CLIP embedding found. Run the Embed job on the CLIP page first.';
            statusEl.className = 'mlt-status';
            return;
          }
          if (data.error === 'no_phash') {
            statusEl.textContent = 'No perceptual hash found. Run the Index job first.';
            statusEl.className = 'mlt-status';
            return;
          }
          if (data.error) {
            statusEl.textContent = data.error;
            statusEl.className = 'mlt-status mlt-error';
            return;
          }
          currentResults = data.results || [];
          if (!currentResults.length) {
            statusEl.textContent = 'No similar photos found at this threshold. Try relaxing it.';
            statusEl.className = 'mlt-status';
          } else {
            statusEl.textContent = currentResults.length + ' similar photo'
              + (currentResults.length === 1 ? '' : 's') + ' found.';
            statusEl.className = 'mlt-status';
            renderResults();
          }
        })
        .catch(function () {
          statusEl.textContent = 'Failed to load similar photos.';
          statusEl.className = 'mlt-status mlt-error';
        });
    }

    function scheduleSimilar() {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(loadSimilar, 400);
    }

    // ── Public API ─────────────────────────────────────────────────────────

    function setReferenceFile(file) {
      referenceFile = file;
      if (headingEl) {
        headingEl.textContent = file ? 'Find similar to loaded image' : 'More like this';
      }
      loadSimilar();
    }

    // Start initial load
    loadSimilar();

    return { setReferenceFile: setReferenceFile };
  }

  return { init: init };
})();
