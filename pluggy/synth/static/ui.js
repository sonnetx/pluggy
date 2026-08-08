/*
 * shared chrome for the pluggy console.
 *
 * loaded synchronously in <head> so the stored theme lands on <html> before
 * first paint -- deferring it would flash light paper on a dark reload.
 *
 * also builds the sidebar's "On this page" jump list by scanning the cards,
 * so each page declares its sections once (in the markup) rather than twice.
 */
(function () {
  var KEY = "pluggy-theme";

  function apply(theme) {
    if (theme === "dark") document.documentElement.setAttribute("data-theme", "dark");
    else document.documentElement.removeAttribute("data-theme");
  }

  function stored() {
    try {
      return localStorage.getItem(KEY) === "dark" ? "dark" : "light";
    } catch (e) {
      return "light";  // private mode / storage disabled
    }
  }

  apply(stored());  // before paint

  var SUN = '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4.2"/>'
    + '<path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.2 5.2l1.4 1.4M17.4 17.4l1.4 1.4'
    + 'M18.8 5.2l-1.4 1.4M6.6 17.4l-1.4 1.4"/></svg>';
  var MOON = '<svg viewBox="0 0 24 24"><path d="M20 14.2A8.2 8.2 0 019.8 4a8.4 8.4 0 100 20 '
    + '8.4 8.4 0 0010.2-9.8z"/></svg>';

  function initTheme() {
    var btn = document.getElementById("themetoggle");
    if (!btn) return;
    function sync() {
      var dark = stored() === "dark";
      btn.innerHTML = dark ? SUN : MOON;
      btn.setAttribute("aria-label", "Switch to " + (dark ? "light" : "dark") + " theme");
      btn.title = btn.getAttribute("aria-label");
    }
    btn.onclick = function () {
      var next = stored() === "dark" ? "light" : "dark";
      try { localStorage.setItem(KEY, next); } catch (e) { /* session-only */ }
      apply(next);
      sync();
    };
    sync();
  }

  /* jump list + scroll spy over the *visible* [data-section] cards. mode
   * switches hide whole card sets, so this is rebuildable rather than
   * built once -- see window.pluggyUI.rebuildSectionNav. */
  var cards = [], links = [], ticking = false;

  function spy() {
    ticking = false;
    var line = 96, active = 0;
    for (var i = 0; i < cards.length; i++) {
      if (cards[i].getBoundingClientRect().top <= line) active = i;
    }
    links.forEach(function (a, i) {
      if (i === active) a.setAttribute("aria-current", "true");
      else a.removeAttribute("aria-current");
    });
  }

  function rebuildSectionNav() {
    var host = document.getElementById("sectionnav");
    if (!host) return;
    host.innerHTML = "";
    cards = [].slice.call(document.querySelectorAll("[data-section]"))
      .filter(function (c) { return c.offsetParent !== null; });  // skip hidden
    links = cards.map(function (card, i) {
      if (!card.id) card.id = "sec-" + i;
      var a = document.createElement("a");
      a.className = "side-item sub";
      a.href = "#" + card.id;
      a.textContent = card.getAttribute("data-section");
      host.appendChild(a);
      return a;
    });
    spy();
  }

  window.addEventListener("scroll", function () {
    if (!ticking) { ticking = true; requestAnimationFrame(spy); }
  }, { passive: true });

  window.pluggyUI = { rebuildSectionNav: rebuildSectionNav };

  document.addEventListener("DOMContentLoaded", function () {
    initTheme();
    rebuildSectionNav();
  });
})();
