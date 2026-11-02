/**
 * Map view: photos arranged by geolocation, with a time dimension.
 *
 * Data comes from /api/map/points as index-aligned arrays sorted by time.
 * Points are bucketed client-side into a screen-space grid at the current
 * zoom level; each non-empty cell shows one representative thumbnail and a
 * count badge.  The timeline either filters photos to a window around the
 * chosen date ("filter") or keeps all photos but shrinks/fades cells whose
 * photos are far from that date ("focus").
 */
(function () {
  'use strict';

  var DAY = 86400;
  var MAX_DRAWER = 400;

  // ── Settings (persisted per browser; storage may be unavailable) ──────────
  function loadSetting(key, fallback) {
    try {
      var v = window.localStorage.getItem('map.' + key);
      return v === null ? fallback : JSON.parse(v);
    } catch (e) { return fallback; }
  }
  // Settings that came from the URL (?from=&to=) are not persisted until the
  // user changes the time controls themselves.
  var ephemeralTime = false;
  function saveSetting(key, value) {
    if (ephemeralTime && (key === 'mode' || key === 'windowDays')) return;
    try { window.localStorage.setItem('map.' + key, JSON.stringify(value)); } catch (e) { /* ignore */ }
  }

  var settings = {
    mode: loadSetting('mode', 'off'),          // off | focus | filter
    windowDays: loadSetting('windowDays', 91),
    size: loadSetting('size', 64),
    tiles: loadSetting('tiles', true),
    fav: loadSetting('fav', false),
    dedupe: loadSetting('dedupe', true),
  };

  var els = {
    stats: document.getElementById('map-stats'),
    fav: document.getElementById('opt-fav'),
    dedupe: document.getElementById('opt-dedupe'),
    windowSel: document.getElementById('opt-window'),
    size: document.getElementById('opt-size'),
    tiles: document.getElementById('opt-tiles'),
    timebar: document.getElementById('map-timebar'),
    time: document.getElementById('map-time'),
    hist: document.getElementById('map-hist'),
    timeLabel: document.getElementById('map-time-label'),
    play: document.getElementById('map-play'),
    empty: document.getElementById('map-empty'),
    notice: document.getElementById('map-notice'),
    drawer: document.getElementById('map-drawer'),
    drawerTitle: document.getElementById('drawer-title'),
    drawerSub: document.getElementById('drawer-sub'),
    drawerGrid: document.getElementById('drawer-grid'),
    drawerZoom: document.getElementById('drawer-zoom'),
    drawerClose: document.getElementById('drawer-close'),
  };

  // The header can wrap onto two lines on narrow windows; size the page to it.
  var headerEl = document.querySelector('header');
  function syncHeaderHeight() {
    if (!headerEl) return;
    document.documentElement.style.setProperty('--header-h', headerEl.offsetHeight + 'px');
  }
  syncHeaderHeight();
  if (headerEl && window.ResizeObserver) {
    new ResizeObserver(function () {
      syncHeaderHeight();
      if (typeof map !== 'undefined') map.invalidateSize();
    }).observe(headerEl);
  }

  // ── Map setup ─────────────────────────────────────────────────────────────
  var map = L.map('map', {
    worldCopyJump: false,
    minZoom: 2,
    maxZoom: 18,
    maxBounds: [[-85, -200], [85, 200]],
    maxBoundsViscosity: 0.8,
    zoomSnap: 1,
  }).setView([30, 10], 2);

  // Offline outline sits in its own pane *below* the tile pane, so online
  // tiles cover it when they load and it shows through when they don't.
  map.createPane('outline');
  map.getPane('outline').style.zIndex = 150;
  map.getPane('outline').classList.add('leaflet-outline-pane');
  fetch('/static/vendor/world/countries-110m.geojson')
    .then(function (r) { return r.json(); })
    .then(function (geo) {
      L.geoJSON(geo, {
        pane: 'outline',
        interactive: false,
        style: { color: '#8aa1b4', weight: 0.7, fillColor: '#f2efe9', fillOpacity: 1 },
      }).addTo(map);
    })
    .catch(function () { /* the map still works without the outline */ });

  var tileErrors = 0;
  var tileLoads = 0;
  var tileLayer = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  });
  tileLayer.on('tileload', function () { tileLoads++; hideNotice(); });
  tileLayer.on('tileerror', function () {
    tileErrors++;
    if (tileLoads === 0 && tileErrors >= 3) {
      showNotice('Map tiles could not be loaded (offline?). Showing the built-in world outline.');
    }
  });
  function applyTiles() {
    if (settings.tiles) { tileLayer.addTo(map); } else { map.removeLayer(tileLayer); hideNotice(); }
  }

  function showNotice(text) { els.notice.textContent = text; els.notice.hidden = false; }
  function hideNotice() { els.notice.hidden = true; }

  // ── Data ──────────────────────────────────────────────────────────────────
  var data = null;     // { n, ids, lat, lon, t, fav, mx, my }
  var tMin = 0;
  var tMax = 0;
  var currentT = 0;
  var targetId = null; // ?asset=ID highlights one photo

  function mercX(lon) { return (lon + 180) / 360; }
  function mercY(lat) {
    var c = Math.max(-85.0511, Math.min(85.0511, lat)) * Math.PI / 180;
    return (1 - Math.log(Math.tan(c) + 1 / Math.cos(c)) / Math.PI) / 2;
  }

  function loadPoints() {
    var q = '?dedupe=' + (settings.dedupe ? '1' : '0') + (settings.fav ? '&favorited=1' : '');
    els.stats.textContent = 'Loading locations…';
    return fetch('/api/map/points' + q)
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function (d) {
        var n = d.count;
        var mx = new Float64Array(n);
        var my = new Float64Array(n);
        for (var i = 0; i < n; i++) {
          mx[i] = mercX(d.lon[i]);
          my[i] = mercY(d.lat[i]);
        }
        data = { n: n, ids: d.ids, lat: d.lat, lon: d.lon, t: d.t, fav: d.fav, mx: mx, my: my };

        tMin = Infinity; tMax = -Infinity;
        for (var j = 0; j < n; j++) {
          var t = d.t[j];
          if (t === null) continue;
          if (t < tMin) tMin = t;
          if (t > tMax) tMax = t;
        }
        if (!isFinite(tMin)) { tMin = 0; tMax = 0; }
        if (!hasRange && (!currentT || currentT < tMin || currentT > tMax)) currentT = tMax;
        histCache = null;

        var pct = d.total_assets ? Math.round(100 * n / d.total_assets) : 0;
        els.stats.textContent = n.toLocaleString() + ' of ' + d.total_assets.toLocaleString() +
          ' photos have a location (' + pct + '%).';
        els.empty.hidden = n > 0;
        clearMarkers();
        drawHistogram();
        syncTimeControls();
        render();
      })
      .catch(function (err) {
        els.stats.textContent = 'Could not load locations.';
        showError('Map: failed to load photo locations (' + err.message + ')');
      });
  }

  // ── Time helpers ──────────────────────────────────────────────────────────
  function sliderToTime(v) { return tMin + (tMax - tMin) * (v / 1000); }
  function timeToSlider(t) { return tMax > tMin ? Math.round(1000 * (t - tMin) / (tMax - tMin)) : 1000; }
  function fmtDate(t) {
    var d = new Date(t * 1000);
    return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
  }
  function fmtMonth(t) {
    return new Date(t * 1000).toLocaleDateString(undefined, { year: 'numeric', month: 'short' });
  }

  /** 1 at the chosen date, falling off with distance; 0 for unknown dates. */
  function timeWeight(t) {
    if (settings.mode === 'off') return 1;
    if (t === null) return settings.mode === 'filter' ? -1 : 0;
    var w = settings.windowDays * DAY;
    var dt = (t - currentT) / w;
    if (settings.mode === 'filter') return Math.abs(dt) <= 1 ? 1 : -1;
    return Math.exp(-0.5 * dt * dt);
  }

  /** Closeness to the chosen date in (0, 1]; used to pick a cell's representative. */
  function proximity(t) {
    if (t === null) return 0;
    var dt = (t - currentT) / (settings.windowDays * DAY);
    return Math.exp(-0.5 * dt * dt);
  }

  function syncTimeControls() {
    els.time.value = timeToSlider(currentT);
    els.timebar.classList.toggle('is-off', settings.mode === 'off');
    if (!data || tMax === 0) { els.timeLabel.textContent = '—'; return; }
    if (settings.mode === 'off') {
      els.timeLabel.textContent = fmtMonth(tMin) + ' – ' + fmtMonth(tMax);
    } else {
      els.timeLabel.textContent = fmtDate(currentT) + ' ± ' + windowLabel();
    }
    drawHistogram();
  }

  function windowLabel() {
    var opt = els.windowSel.options[els.windowSel.selectedIndex];
    return opt ? opt.textContent.replace('± ', '') : settings.windowDays + ' days';
  }

  // ── Histogram of photos over time (with the active window shaded) ─────────
  var histCache = null; // { bins, counts, max }

  function drawHistogram() {
    var canvas = els.hist;
    var dpr = window.devicePixelRatio || 1;
    var w = canvas.clientWidth;
    var h = canvas.clientHeight;
    if (!w || !h) return;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!data || tMax <= tMin) return;

    var bins = Math.max(10, Math.min(400, Math.floor(w / 3)));
    var span = tMax - tMin;
    // Binning touches every point, so only redo it when the data or width changes.
    if (!histCache || histCache.bins !== bins) {
      var cnt = new Array(bins).fill(0);
      for (var i = 0; i < data.n; i++) {
        var t = data.t[i];
        if (t === null) continue;
        cnt[Math.min(bins - 1, Math.floor((t - tMin) / span * bins))]++;
      }
      histCache = { bins: bins, counts: cnt, max: Math.max.apply(null, cnt) || 1 };
    }
    var counts = histCache.counts;
    var maxC = histCache.max;

    if (settings.mode !== 'off') {
      var ww = settings.windowDays * DAY;
      var x0 = (currentT - ww - tMin) / span * w;
      var x1 = (currentT + ww - tMin) / span * w;
      ctx.fillStyle = 'rgba(26, 92, 219, 0.12)';
      ctx.fillRect(x0, 0, Math.max(2, x1 - x0), h);
    }
    var bw = w / bins;
    for (var b = 0; b < bins; b++) {
      if (!counts[b]) continue;
      var bh = Math.max(1, Math.sqrt(counts[b] / maxC) * (h - 2));
      var mid = tMin + (b + 0.5) / bins * span;
      ctx.fillStyle = settings.mode === 'off' || timeWeight(mid) > 0.05 ? '#1a5cdb' : '#b8c4d8';
      ctx.fillRect(b * bw, h - bh, Math.max(1, bw - 0.5), bh);
    }
    // Year ticks
    ctx.fillStyle = '#888';
    ctx.font = '10px system-ui, sans-serif';
    var y0 = new Date(tMin * 1000).getFullYear() + 1;
    var y1 = new Date(tMax * 1000).getFullYear();
    var step = Math.max(1, Math.ceil((y1 - y0 + 1) / Math.max(1, Math.floor(w / 45))));
    for (var y = y0; y <= y1; y += step) {
      var ty = Date.UTC(y, 0, 1) / 1000;
      var x = (ty - tMin) / span * w;
      ctx.fillRect(x, 0, 1, 4);
      ctx.fillText(String(y), x + 2, 9);
    }
  }

  // ── Grid clustering & marker rendering ────────────────────────────────────
  var markers = new Map();   // cell key -> { marker, el, img, badge, id, info }
  var lastCells = new Map(); // cell key -> cell info from the last render
  var lastZoom = null;
  var selectedKey = null;

  function clearMarkers() {
    markers.forEach(function (m) { map.removeLayer(m.marker); });
    markers.clear();
    lastCells.clear();
    selectedKey = null; // cell keys change with zoom / thumbnail size
  }

  function cellSize() { return settings.size + 10; }

  function render() {
    if (!data) return;
    var z = map.getZoom();
    if (z !== lastZoom) { clearMarkers(); lastZoom = z; }
    var scale = 256 * Math.pow(2, z);
    var pb = map.getPixelBounds();
    var cs = cellSize();
    var minX = pb.min.x - cs, maxX = pb.max.x + cs;
    var minY = pb.min.y - cs, maxY = pb.max.y + cs;

    var cells = new Map();
    for (var i = 0; i < data.n; i++) {
      var px = data.mx[i] * scale;
      if (px < minX || px > maxX) continue;
      var py = data.my[i] * scale;
      if (py < minY || py > maxY) continue;
      var wgt = timeWeight(data.t[i]);
      if (wgt < 0) continue;

      var key = Math.floor(px / cs) + ':' + Math.floor(py / cs);
      var c = cells.get(key);
      if (!c) {
        c = { count: 0, sx: 0, sy: 0, best: i, bestScore: -Infinity, maxW: 0 };
        cells.set(key, c);
      }
      c.count++;
      c.sx += px; c.sy += py;
      if (wgt > c.maxW) c.maxW = wgt;
      // Representative: closest in time when a date is chosen, otherwise newest;
      // favorites win ties.  Points are sorted by time, so ">=" keeps the newest.
      var score = (settings.mode === 'off' ? 0 : proximity(data.t[i]) * 10) + data.fav[i] * 0.5 +
        (data.ids[i] === targetId ? 100 : 0);
      if (score >= c.bestScore) { c.bestScore = score; c.best = i; }
    }

    // Remove stale markers
    markers.forEach(function (m, key) {
      if (!cells.has(key)) { map.removeLayer(m.marker); markers.delete(key); }
    });

    cells.forEach(function (c, key) {
      var latlng = map.unproject([c.sx / c.count, c.sy / c.count], z);
      var id = data.ids[c.best];
      var f = settings.mode === 'focus' ? c.maxW : 1;
      var size = Math.round(settings.size * (0.3 + 0.7 * f));
      var m = markers.get(key);
      if (!m) {
        var el = document.createElement('div');
        el.className = 'mc';
        var img = document.createElement('img');
        img.loading = 'lazy';
        img.alt = '';
        el.appendChild(img);
        var badge = document.createElement('span');
        badge.className = 'mc-count';
        el.appendChild(badge);
        var marker = L.marker(latlng, {
          icon: L.divIcon({ className: 'map-cell', html: el, iconSize: [0, 0], iconAnchor: [0, 0] }),
          keyboard: false,
        });
        m = { marker: marker, el: el, img: img, badge: badge, id: null, key: key };
        marker.on('click', (function (k) { return function () { openDrawer(k); }; })(key));
        markers.set(key, m);
        marker.addTo(map);
      } else {
        m.marker.setLatLng(latlng);
      }
      if (m.id !== id) {
        m.id = id;
        m.img.src = '/thumbs/' + id;
      }
      m.el.style.width = size + 'px';
      m.el.style.height = size + 'px';
      m.el.style.opacity = settings.mode === 'focus' ? (0.35 + 0.65 * f).toFixed(2) : '1';
      m.marker.setZIndexOffset(Math.round(f * 1000) + (id === targetId ? 5000 : 0));
      m.badge.textContent = c.count > 1 ? (c.count > 9999 ? Math.round(c.count / 1000) + 'k' : c.count) : '';
      m.badge.style.display = c.count > 1 ? '' : 'none';
      m.el.classList.toggle('is-selected', key === selectedKey);
      m.el.classList.toggle('is-target', id === targetId);
    });
    lastCells = cells;
  }

  var renderQueued = false;
  function scheduleRender() {
    if (renderQueued) return;
    renderQueued = true;
    window.requestAnimationFrame(function () { renderQueued = false; render(); });
  }

  // ── Drawer: photos in a clicked cell ──────────────────────────────────────
  var drawerMembers = [];

  function cellMembers(key) {
    var z = map.getZoom();
    var scale = 256 * Math.pow(2, z);
    var cs = cellSize();
    var out = [];
    for (var i = 0; i < data.n; i++) {
      var k = Math.floor(data.mx[i] * scale / cs) + ':' + Math.floor(data.my[i] * scale / cs);
      if (k !== key) continue;
      var wgt = timeWeight(data.t[i]);
      if (wgt < 0) continue;
      out.push(i);
    }
    if (settings.mode === 'off') {
      out.sort(function (a, b) { return (data.t[b] || 0) - (data.t[a] || 0); });
    } else {
      out.sort(function (a, b) {
        var da = data.t[a] === null ? Infinity : Math.abs(data.t[a] - currentT);
        var db = data.t[b] === null ? Infinity : Math.abs(data.t[b] - currentT);
        return da - db;
      });
    }
    return out;
  }

  function openDrawer(key) {
    selectedKey = key;
    drawerMembers = cellMembers(key);
    var n = drawerMembers.length;
    els.drawerTitle.textContent = n.toLocaleString() + ' photo' + (n === 1 ? '' : 's');
    var ts = drawerMembers.map(function (i) { return data.t[i]; }).filter(function (t) { return t !== null; });
    if (ts.length) {
      var a = Math.min.apply(null, ts.slice(0, 50000));
      var b = Math.max.apply(null, ts.slice(0, 50000));
      els.drawerSub.textContent = fmtMonth(a) === fmtMonth(b) ? fmtMonth(a) : fmtMonth(a) + ' – ' + fmtMonth(b);
    } else {
      els.drawerSub.textContent = 'No dates';
    }
    if (settings.mode !== 'off') els.drawerSub.textContent += ' · nearest to ' + fmtDate(currentT) + ' first';

    var frag = document.createDocumentFragment();
    drawerMembers.slice(0, MAX_DRAWER).forEach(function (i) {
      var id = data.ids[i];
      var card = document.createElement('div');
      card.className = 'card';
      card.dataset.lbId = id;
      card.dataset.lbSrc = '/image/' + id;
      card.dataset.lbTitle = data.t[i] !== null ? fmtDate(data.t[i]) : 'Photo ' + id;
      var img = document.createElement('img');
      img.loading = 'lazy';
      img.src = '/thumbs/' + id;
      img.alt = '';
      card.appendChild(img);
      var cap = document.createElement('span');
      cap.textContent = (data.fav[i] ? '⭐ ' : '') + (data.t[i] !== null ? fmtDate(data.t[i]) : '');
      card.appendChild(cap);
      frag.appendChild(card);
    });
    if (n > MAX_DRAWER) {
      var more = document.createElement('div');
      more.className = 'map-drawer-more';
      more.textContent = 'Showing ' + MAX_DRAWER + ' of ' + n.toLocaleString() + '. Zoom in to narrow down.';
      frag.appendChild(more);
    }
    els.drawerGrid.replaceChildren(frag);
    els.drawerGrid.scrollTop = 0;
    els.drawer.hidden = false;
    render();
  }

  function closeDrawer() {
    els.drawer.hidden = true;
    selectedKey = null;
    drawerMembers = [];
    render();
  }

  els.drawerClose.addEventListener('click', closeDrawer);
  els.drawerZoom.addEventListener('click', function () {
    if (!drawerMembers.length) return;
    var pts = drawerMembers.map(function (i) { return [data.lat[i], data.lon[i]]; });
    closeDrawer();
    map.fitBounds(L.latLngBounds(pts), { padding: [60, 60], maxZoom: 16 });
  });

  SharedLightbox.init({ gridEl: els.drawerGrid, cardSelector: '.card[data-lb-id]' });

  // ── Controls ──────────────────────────────────────────────────────────────
  function setMode(mode) {
    settings.mode = mode;
    saveSetting('mode', mode);
    document.querySelectorAll('input[name=time-mode]').forEach(function (r) { r.checked = r.value === mode; });
    syncTimeControls();
    scheduleRender();
  }

  document.querySelectorAll('input[name=time-mode]').forEach(function (r) {
    r.checked = r.value === settings.mode;
    r.addEventListener('change', function () {
      if (!r.checked) return;
      if (ephemeralTime && els.windowSel.selectedIndex >= 5) {
        // Leaving a deep-link window: fall back to a regular window option.
        settings.windowDays = 91;
        els.windowSel.value = '91';
      }
      ephemeralTime = false;
      saveSetting('windowDays', settings.windowDays);
      setMode(r.value);
    });
  });

  els.time.addEventListener('input', function () {
    currentT = sliderToTime(Number(els.time.value));
    if (settings.mode === 'off') { setMode('focus'); return; }
    syncTimeControls();
    scheduleRender();
  });

  els.windowSel.value = String(settings.windowDays);
  if (els.windowSel.value !== String(settings.windowDays)) {
    // Stored value is not one of the options (e.g. an old deep-link window).
    settings.windowDays = 91;
    els.windowSel.value = '91';
  }
  els.windowSel.addEventListener('change', function () {
    settings.windowDays = Number(els.windowSel.value);
    ephemeralTime = false;
    saveSetting('mode', settings.mode);
    saveSetting('windowDays', settings.windowDays);
    syncTimeControls();
    scheduleRender();
  });

  els.size.value = settings.size;
  els.size.addEventListener('input', function () {
    settings.size = Number(els.size.value);
    saveSetting('size', settings.size);
    clearMarkers();
    scheduleRender();
  });

  els.tiles.checked = settings.tiles;
  els.tiles.addEventListener('change', function () {
    settings.tiles = els.tiles.checked;
    saveSetting('tiles', settings.tiles);
    applyTiles();
  });

  els.fav.checked = settings.fav;
  els.fav.addEventListener('change', function () {
    settings.fav = els.fav.checked;
    saveSetting('fav', settings.fav);
    closeDrawer();
    loadPoints();
  });

  els.dedupe.checked = settings.dedupe;
  els.dedupe.addEventListener('change', function () {
    settings.dedupe = els.dedupe.checked;
    saveSetting('dedupe', settings.dedupe);
    closeDrawer();
    loadPoints();
  });

  // Play through time: advance by a third of the window per frame.
  var playTimer = null;
  function stopPlay() {
    if (playTimer) { clearInterval(playTimer); playTimer = null; }
    els.play.textContent = '▶';
  }
  els.play.addEventListener('click', function () {
    if (playTimer) { stopPlay(); return; }
    if (!data || tMax <= tMin) return;
    if (settings.mode === 'off') setMode('focus');
    if (currentT >= tMax) currentT = tMin;
    els.play.textContent = '⏸';
    playTimer = setInterval(function () {
      currentT = Math.min(tMax, currentT + settings.windowDays * DAY / 3);
      syncTimeControls();
      scheduleRender();
      if (currentT >= tMax) stopPlay();
    }, 350);
  });

  map.on('moveend zoomend', scheduleRender);
  window.addEventListener('resize', function () { drawHistogram(); });

  // ── Start ─────────────────────────────────────────────────────────────────
  // URL parameters (not persisted):
  //   ?asset=ID              centre on one photo and highlight it
  //   ?from=TS&to=TS         Unix seconds; show only photos in that range and fit to them
  var params = new URLSearchParams(window.location.search);
  if (params.get('asset')) targetId = Number(params.get('asset'));
  var rangeFrom = Number(params.get('from'));
  var rangeTo = Number(params.get('to'));
  var hasRange = rangeFrom > 0 && rangeTo >= rangeFrom;
  if (hasRange) {
    ephemeralTime = true;
    settings.mode = 'filter';
    // Keep the exact interval: the window is a (possibly fractional) half-span in days.
    settings.windowDays = Math.max(1, rangeTo - rangeFrom) / 2 / DAY;
    currentT = (rangeFrom + rangeTo) / 2;
    var opt = document.createElement('option');
    opt.value = String(settings.windowDays);
    var halfDays = settings.windowDays;
    opt.textContent = '± ' + (halfDays >= 1
      ? Math.round(halfDays * 10) / 10 + ' days'
      : Math.max(1, Math.round(halfDays * 24 * 60)) + ' min');
    els.windowSel.appendChild(opt);
    els.windowSel.value = opt.value;
    document.querySelectorAll('input[name=time-mode]').forEach(function (r) { r.checked = r.value === 'filter'; });
  }

  applyTiles();
  loadPoints().then(function () {
    // A duplicate that isn't its group's representative is hidden by dedupe;
    // show duplicates for this visit (not saved) so the linked photo appears.
    if (data && targetId !== null && settings.dedupe && data.ids.indexOf(targetId) < 0) {
      settings.dedupe = false;
      els.dedupe.checked = false;
      return loadPoints();
    }
  }).then(function () {
    if (!data || data.n === 0) return;
    if (targetId !== null) {
      var idx = data.ids.indexOf(targetId);
      if (idx >= 0) {
        if (data.t[idx] !== null) { currentT = data.t[idx]; syncTimeControls(); }
        map.setView([data.lat[idx], data.lon[idx]], 15);
        return;
      }
      showNotice('That photo has no location (or is hidden by the current filters).');
    }
    var latMin = 90, latMax = -90, lonMin = 180, lonMax = -180;
    for (var i = 0; i < data.n; i++) {
      if (hasRange && timeWeight(data.t[i]) < 0) continue;
      if (data.lat[i] < latMin) latMin = data.lat[i];
      if (data.lat[i] > latMax) latMax = data.lat[i];
      if (data.lon[i] < lonMin) lonMin = data.lon[i];
      if (data.lon[i] > lonMax) lonMax = data.lon[i];
    }
    if (latMin > latMax) return; // nothing in the requested range
    map.fitBounds([[latMin, lonMin], [latMax, lonMax]], { padding: [40, 40], maxZoom: 12 });
  });
})();
