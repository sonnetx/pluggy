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

  // feather icons (MIT). both are built on a shape centred at (12,12) in a
  // 24x24 box, so they sit true in the button and clear the stroke width.
  var SUN = '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="5"/>'
    + '<path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42'
    + 'M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>';
  var MOON = '<svg viewBox="0 0 24 24">'
    + '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';

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
