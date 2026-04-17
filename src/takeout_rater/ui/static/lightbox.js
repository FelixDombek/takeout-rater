/**
 * Shared lightbox module for takeout-rater.
 *
 * Usage:
 *   SharedLightbox.init({
 *     gridEl:       document.getElementById('photo-grid'), // container of cards
 *     cardSelector: '.card[data-lb-id]',                  // selector for clickable cards
 *     onNextPage:   function(callback) { ... },           // optional: load more, then call callback
 *   });
 *
 * Each card element must have:
 *   data-lb-id    – asset ID
 *   data-lb-src   – thumbnail URL
 *   data-lb-title – display title
 *
 * The page must contain a #lightbox element with the structure from lightbox.html.
 */
window.SharedLightbox = (function () {
  'use strict';

  function init(opts) {
    var options      = opts || {};
    var gridEl       = options.gridEl || null;
    var cardSelector = options.cardSelector || '[data-lb-id]';
    var onNextPage   = options.onNextPage || null;

    var lightbox  = document.getElementById('lightbox');
    var lbImg     = document.getElementById('lb-img');
    var lbTitle   = document.getElementById('lb-title');
    var lbLink    = document.getElementById('lb-link');
    var lbCounter = document.getElementById('lb-counter');
    var lbDetails = document.getElementById('lb-details');

    if (!lightbox) return {};

    var lbIndex = 0;
    var lbDetailAbort = null;

    function getCards() {
      var root = gridEl || document;
      return Array.from(root.querySelectorAll(cardSelector));
    }

    function highlightJson(pre) {
      var text = pre.textContent;
      var html = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      html = html.replace(
        /("(?:\\.|[^"\\])*")(\s*:)|("(?:\\.|[^"\\])*")|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|(\btrue\b|\bfalse\b)|(\bnull\b)/g,
        function (m, key, colon, str, num, bool, nil) {
          if (key && colon) return '<span class="json-key">' + key + '</span>' + colon;
          if (str)  return '<span class="json-string">' + str + '</span>';
          if (num)  return '<span class="json-number">' + num + '</span>';
          if (bool) return '<span class="json-bool">' + bool + '</span>';
          if (nil)  return '<span class="json-null">' + nil + '</span>';
          return m;
        }
      );
      pre.innerHTML = html;
    }

    function lbLoadClipWords(panel) {
      var assetId = panel.dataset.assetId;
      var loadingEl = panel.querySelector('.lb-clip-loading');
      var emptyEl   = panel.querySelector('.lb-clip-empty');
      if (loadingEl) loadingEl.style.display = '';
      if (emptyEl)   emptyEl.style.display   = 'none';
      fetch('/api/assets/' + assetId + '/clip-words?top_k=30')
        .then(function (r) { return r.json(); })
        .then(function (data) {
          if (loadingEl) loadingEl.style.display = 'none';
          if (!data.words || data.words.length === 0) {
            if (emptyEl) emptyEl.style.display = '';
            return;
          }
          panel.dataset.loaded = '1';
          var maxScore = data.words[0].score || 1;
          var cloud = document.createElement('div');
          cloud.className = 'lb-clip-cloud';
          data.words.forEach(function (w) {
            var tag = document.createElement('span');
            tag.className = 'lb-clip-tag' + (w.user_tag ? ' lb-clip-tag--user' : '');
            var rel = w.score / maxScore;
            tag.style.opacity    = (0.4 + 0.6 * rel).toFixed(2);
            tag.style.fontSize   = (0.65 + 0.3 * rel).toFixed(2) + 'rem';
            tag.title            = (w.user_tag ? 'custom tag — ' : '') + 'score: ' + w.score.toFixed(4);
            tag.textContent      = w.word;
            cloud.appendChild(tag);
          });
          panel.appendChild(cloud);
        })
        .catch(function () {
          if (loadingEl) loadingEl.style.display = 'none';
          if (emptyEl)   emptyEl.style.display   = '';
        });
    }

    function loadDetails(assetId) {
      if (lbDetailAbort) lbDetailAbort.abort();
      lbDetailAbort = new AbortController();
      lbDetails.innerHTML = '<div class="lb-details-loading">Loading\u2026</div>';
      fetch('/assets/' + assetId + '?partial=1', { signal: lbDetailAbort.signal })
        .then(function (r) {
          if (!r.ok) throw new Error('HTTP ' + r.status);
          return r.text();
        })
        .then(function (html) {
          lbDetails.innerHTML = html;
          lbDetails.querySelectorAll('.json-highlight').forEach(highlightJson);

          // Draw pHash canvas (inline scripts don't run via innerHTML).
          lbDetails.querySelectorAll('.lb-phash-canvas[data-phash]').forEach(function (canvas) {
            var hex = canvas.dataset.phash;
            if (!hex) return;
            var ctx = canvas.getContext('2d');
            var BITS = 256, SIDE = 16, BLOCK = 4;
            var bigInt = BigInt('0x' + hex);
            for (var idx = 0; idx < BITS; idx++) {
              var bit = Number((bigInt >> BigInt(idx)) & 1n);
              var row = Math.floor(idx / SIDE);
              var col = idx % SIDE;
              ctx.fillStyle = bit ? '#ffffff' : '#1a1a2e';
              ctx.fillRect(col * BLOCK, row * BLOCK, BLOCK, BLOCK);
            }
          });

          // Fetch and draw CLIP embedding grid (32×24 grayscale).
          lbDetails.querySelectorAll('.lb-embed-wrap[data-asset-id]').forEach(function (wrap) {
            var assetId = wrap.dataset.assetId;
            var canvas = wrap.querySelector('.lb-embed-canvas');
            if (!assetId || !canvas) return;
            fetch('/api/assets/' + assetId + '/clip-embedding')
              .then(function (r) { return r.json(); })
              .then(function (data) {
                var COLS = 32, ROWS = 24, BLOCK = 4;
                if (!data.values || data.values.length !== COLS * ROWS) return;
                wrap.style.display = '';
                var ctx = canvas.getContext('2d');
                for (var i = 0; i < data.values.length; i++) {
                  var gray = Math.round(data.values[i] * 255);
                  var col = i % COLS;
                  var row = Math.floor(i / COLS);
                  ctx.fillStyle = 'rgb(' + gray + ',' + gray + ',' + gray + ')';
                  ctx.fillRect(col * BLOCK, row * BLOCK, BLOCK, BLOCK);
                }
              })
              .catch(function () {});
          });

          // Wire tab buttons (injected via innerHTML don't have event listeners).
          lbDetails.querySelectorAll('.lb-tab-btn').forEach(function (btn) {
            btn.addEventListener('click', function () {
              var tabName = btn.dataset.tab;
              lbDetails.querySelectorAll('.lb-tab-btn').forEach(function (b) {
                var active = b.dataset.tab === tabName;
                b.classList.toggle('active', active);
                b.setAttribute('aria-selected', active ? 'true' : 'false');
              });
              lbDetails.querySelectorAll('.lb-tab-panel').forEach(function (p) {
                p.classList.toggle('active', p.dataset.panel === tabName);
              });
              if (tabName === 'clip') {
                var panel = lbDetails.querySelector('[data-panel="clip"]');
                if (panel && !panel.dataset.loaded) {
                  lbLoadClipWords(panel);
                }
              }
            });
          });
        })
        .catch(function (err) {
          if (err.name !== 'AbortError') {
            lbDetails.innerHTML = '<div class="lb-details-loading">Could not load details.</div>';
          }
        });
    }

    function openLightbox(index) {
      var cards = getCards();
      if (!cards.length) return;
      lbIndex = Math.max(0, Math.min(index, cards.length - 1));
      var card = cards[lbIndex];

      var prev = document.querySelector('.card--active');
      if (prev) prev.classList.remove('card--active');
      card.classList.add('card--active');

      lbImg.src           = card.dataset.lbSrc   || '';
      lbImg.alt           = card.dataset.lbTitle  || '';
      lbTitle.textContent = card.dataset.lbTitle  || '';
      lbLink.href         = '/assets/' + card.dataset.lbId;
      lbCounter.textContent = (lbIndex + 1) + ' / ' + cards.length;

      lightbox.style.display = 'flex';
      lightbox.setAttribute('aria-hidden', 'false');
      document.body.style.overflow = 'hidden';
      lbImg.focus();
      loadDetails(card.dataset.lbId);
    }

    function lbClose() {
      if (lbDetailAbort) { lbDetailAbort.abort(); lbDetailAbort = null; }
      var active = document.querySelector('.card--active');
      if (active) active.classList.remove('card--active');
      lightbox.style.display = 'none';
      lightbox.setAttribute('aria-hidden', 'true');
      document.body.style.overflow = '';
      lbImg.src = '';
      lbDetails.innerHTML = '<div class="lb-details-loading">Loading\u2026</div>';
    }

    function lbPrev() {
      if (lbIndex > 0) openLightbox(lbIndex - 1);
    }

    function lbNext() {
      var cards = getCards();
      if (lbIndex < cards.length - 1) {
        openLightbox(lbIndex + 1);
      } else if (onNextPage) {
        var currentCount = cards.length;
        onNextPage(function () {
          var newCards = getCards();
          if (newCards.length > currentCount) openLightbox(currentCount);
        });
      }
    }

    document.getElementById('lb-close').addEventListener('click', lbClose);
    document.getElementById('lb-prev').addEventListener('click', lbPrev);
    document.getElementById('lb-next').addEventListener('click', lbNext);

    if (gridEl) {
      gridEl.addEventListener('click', function (e) {
        var card = e.target.closest(cardSelector);
        if (!card) return;
        e.preventDefault();
        var cards = getCards();
        openLightbox(cards.indexOf(card));
      });
    }

    lightbox.addEventListener('click', function (e) {
      if (e.target === lightbox) lbClose();
    });

    document.addEventListener('keydown', function (e) {
      if (lightbox.style.display !== 'flex') return;
      if (e.key === 'Escape')     { e.preventDefault(); lbClose(); }
      if (e.key === 'ArrowLeft')  { e.preventDefault(); lbPrev(); }
      if (e.key === 'ArrowRight') { e.preventDefault(); lbNext(); }
    });

    return { open: openLightbox, close: lbClose };
  }

  return { init: init };
})();
