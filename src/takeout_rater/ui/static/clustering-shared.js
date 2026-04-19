(function (global) {
  'use strict';

  function deleteSavedRun(options) {
    if (!confirm(options.confirmMessage)) return;

    var card = document.getElementById(options.cardId);
    var statusEl = document.getElementById(options.statusId);
    var btn = card ? card.querySelector('.run-actions .btn') : null;

    if (btn) btn.disabled = true;
    if (statusEl) {
      statusEl.textContent = 'Deleting\u2026';
      statusEl.style.color = '#888';
    }

    fetch(options.url, { method: 'DELETE' })
      .then(function(r) {
        if (!r.ok) {
          return r.json().then(function(d) {
            throw new Error(d.detail || r.statusText);
          });
        }
        return r.json();
      })
      .then(function() {
        if (card) {
          card.style.transition = 'opacity 0.3s';
          card.style.opacity = '0';
          setTimeout(function() { card.remove(); }, 300);
        }
      })
      .catch(function(err) {
        if (statusEl) {
          statusEl.textContent = '\u2717 ' + err.message;
          statusEl.style.color = '#c0392b';
        }
        if (btn) btn.disabled = false;
      });
  }

  global.deleteSavedRun = deleteSavedRun;
}(window));
