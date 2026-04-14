/**
 * jobs-shared.js – shared progress-bar and throughput/ETA helpers.
 *
 * Exposes two globals:
 *
 *   formatJobEta(seconds)
 *     Format a remaining-time estimate as a short human-readable string
 *     (e.g. "1m 23s", "45s").
 *
 *   appendJobEtaToMessage(msg, processed, total, stats)
 *     Given a base status message and an ETA tracking object
 *     {startTime, startProcessed}, appends a "N/s · ETA Xs" suffix when
 *     there is enough data.  Returns the (possibly extended) message string.
 *
 *   updateJobEtaStats(stats, processed)
 *     Return a new (or updated) stats object anchored to now.  Pass null for
 *     stats to create a fresh anchor.  Re-anchors automatically when processed
 *     has not advanced beyond the previous anchor.
 *
 *   setProgressBar(wrapId, barId, pct)
 *     Show/hide a progress bar identified by element IDs.  Pass null for pct
 *     to hide the bar.
 */
(function (global) {
  'use strict';

  function formatJobEta(seconds) {
    if (seconds < 60) return seconds + 's';
    var m = Math.floor(seconds / 60);
    var s = seconds % 60;
    return m + 'm\u202f' + (s < 10 ? '0' : '') + s + 's';
  }

  /**
   * Append throughput and ETA information to a status message.
   *
   * @param {string} msg - Base message.
   * @param {number} processed - Items processed so far.
   * @param {number} total - Total items (0 if unknown).
   * @param {object|null} stats - Anchor object {startTime, startProcessed}.
   * @returns {string} Message with optional " · N/s · ETA Xs" suffix.
   */
  function appendJobEtaToMessage(msg, processed, total, stats) {
    if (!stats || total <= 0) return msg;
    var elapsed = (Date.now() - stats.startTime) / 1000;
    var done = processed - stats.startProcessed;
    if (elapsed >= 1 && done > 0) {
      var rate = done / elapsed;
      var eta = Math.round((total - processed) / rate);
      msg += '\u2002\u00b7\u2002' + rate.toFixed(1) + '/s';
      if (eta > 0) msg += '\u2002\u00b7\u2002ETA\u202f' + formatJobEta(eta);
    }
    return msg;
  }

  /**
   * Update (or create) a throughput anchor object.
   *
   * Re-anchors when processed has not advanced beyond the previous anchor,
   * which avoids inflated rate estimates during job-phase transitions where
   * the processed counter resets.
   *
   * @param {object|null} stats - Existing anchor, or null for a fresh one.
   * @param {number} processed - Current processed count.
   * @returns {object} Updated anchor {startTime, startProcessed}.
   */
  function updateJobEtaStats(stats, processed) {
    if (!stats || processed <= stats.startProcessed) {
      return { startTime: Date.now(), startProcessed: processed };
    }
    return stats;
  }

  /**
   * Update a progress bar by element IDs.
   *
   * @param {string} wrapId - ID of the wrapper element.
   * @param {string} barId  - ID of the inner bar element.
   * @param {number|null} pct - Percentage (0–100) or null to hide.
   */
  function setProgressBar(wrapId, barId, pct) {
    var wrap = document.getElementById(wrapId);
    var bar  = document.getElementById(barId);
    if (!wrap || !bar) return;
    if (pct === null || pct === undefined) {
      wrap.className = 'progress-bar-wrap';
    } else {
      wrap.className = 'progress-bar-wrap visible';
      bar.style.width = Math.min(100, Math.max(0, pct)) + '%';
    }
  }

  global.formatJobEta       = formatJobEta;
  global.appendJobEtaToMessage = appendJobEtaToMessage;
  global.updateJobEtaStats  = updateJobEtaStats;
  global.setProgressBar     = setProgressBar;
}(window));
