/* Developed by Carpathian, LLC. Distribution Not Authorized. */
/* veritate_mri/static/tutorial.js */

// Data-driven walkthrough overlay. Spotlights a real target element (darkening
// everything else), shows a tooltip card with back/next/skip. Self-contained:
// wires via DOM + /settings, no index.js edits. Blue (--accent) themed.
// window.Tutorial.start() runs it; auto-starts when tutorial_enabled && !tutorial_completed.

(function () {
  "use strict";

  const SETTINGS_URL = "/settings";
  const PAD = 8;
  const CARD_GAP = 14;
  const CARD_W = 330;

  // step: { tab, target, title, body }
  // Every step targets a real element so the spotlight always has something to
  // frame. `target` may be a list; the first match found is used.
  const STEPS = [
    {
      tab: "generation",
      target: ['.tabs'],
      title: "Welcome to Veritate",
      body: "This is the dashboard for training and running byte-level models on your own hardware. These tabs are the whole platform. Let me show you each one in about a minute.",
    },
    {
      tab: "generation",
      target: ['#prompt'],
      title: "1. Generation",
      body: "Talk to a loaded model right here. Type a prompt, send it, and watch the model generate a response token by token.",
    },
    {
      tab: "learning",
      target: ['.tab[data-tab="learning"]'],
      title: "2. Models",
      body: "Every checkpoint you have trained lives here. Inspect a model's size and steps, then load it so the Generation tab can use it.",
    },
    {
      tab: "training",
      target: ['#trainFlowOpenBtn', '#trainStep1Row'],
      title: "3. Training starts here",
      body: "Click 'Choose action' to begin: train a new model from scratch, continue an existing one, or distill from a teacher. The flow walks you through trainer, then settings, then start.",
    },
    {
      tab: "training",
      target: ['#trainAutoOptimizeHelp', '#trainEstimateRow'],
      title: "Auto optimize",
      body: "Not sure what batch size or learning rate to pick? Auto optimize fits the settings to your detected hardware. Review the values, then start the run.",
    },
    {
      tab: "wiki",
      target: ['.tab[data-tab="wiki"]'],
      title: "4. Wiki",
      body: "Build notes and platform docs. After any update, check here for what changed and anything you need to do.",
    },
    {
      tab: "logs",
      target: ['.tab[data-tab="logs"]'],
      title: "5. Logs",
      body: "Live platform and training output. When a run stalls or errors, this is the first place to look.",
    },
    {
      tab: "settings",
      target: ['#minimalModeRow'],
      title: "6. Power save",
      body: "Low on memory? Power save re-launches the dashboard without the heavy inference brain and background threads, freeing several GB. Your training keeps running.",
    },
    {
      tab: "settings",
      target: ['#trainersSyncRow'],
      title: "Download trainers",
      body: "Trainers are the model recipes. This panel pulls them from the public trainers repo: click 'check', then 'update' to install or refresh them. They then appear in the Training tab's trainer picker.",
    },
    {
      tab: "settings",
      target: ['#corpusLibraryRow'],
      title: "Download corpuses",
      body: "A corpus is the training data a model reads. Click 'browse' to open the library and install one. It downloads into trainers/corpus/ and shows up in the Training tab's data picker.",
    },
    {
      tab: "settings",
      target: ['#tutorialRow'],
      title: "You are all set",
      body: "Restart this tour or turn it off any time from here. That is the whole platform. Happy training.",
    },
  ];

  let idx = 0;
  let root = null;

  function get_settings() {
    return fetch(SETTINGS_URL).then(r => r.json()).catch(() => ({}));
  }

  function post_settings(patch) {
    return fetch(SETTINGS_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }).catch(() => {});
  }

  function set_enabled(on) { return post_settings({ tutorial_enabled: !!on }); }

  function switch_tab(name) {
    const tab = document.querySelector(`.tab[data-tab="${name}"]`);
    if (tab) tab.click();
  }

  function find_target(step) {
    const list = Array.isArray(step.target) ? step.target : (step.target ? [step.target] : []);
    for (const sel of list) {
      const el = document.querySelector(sel);
      if (el && el.getBoundingClientRect().width > 0) return el;
    }
    return null;
  }

  function build_root() {
    const el = document.createElement("div");
    el.id = "tutorialOverlay";
    const dots = STEPS.map(() => '<span class="tut-dot"></span>').join("");
    el.innerHTML =
      '<div class="tut-spot"></div>' +
      '<div class="tut-card">' +
      '<div class="tut-title"></div>' +
      '<div class="tut-body"></div>' +
      '<div class="tut-foot">' +
      `<span class="tut-progress">${dots}</span>` +
      '<span class="tut-btns">' +
      '<button type="button" class="tut-skip">skip</button>' +
      '<button type="button" class="tut-back">back</button>' +
      '<button type="button" class="tut-next">next</button>' +
      "</span></div></div>";
    document.body.appendChild(el);
    el.querySelector(".tut-skip").addEventListener("click", finish);
    el.querySelector(".tut-back").addEventListener("click", () => go(idx - 1));
    el.querySelector(".tut-next").addEventListener("click", () => go(idx + 1));
    window.addEventListener("resize", render_passive);
    return el;
  }

  function place_card(card, rect) {
    if (!rect) {
      card.style.left = `${(window.innerWidth - CARD_W) / 2}px`;
      card.style.top = `${Math.max(70, window.innerHeight / 2 - card.offsetHeight / 2)}px`;
      return;
    }
    // Prefer below the target, else above, clamped to viewport.
    let top = rect.bottom + CARD_GAP;
    if (top + card.offsetHeight > window.innerHeight - 12) {
      top = rect.top - card.offsetHeight - CARD_GAP;
    }
    if (top < 12) top = 12;
    let left = rect.left + rect.width / 2 - CARD_W / 2;
    if (left + CARD_W > window.innerWidth - 12) left = window.innerWidth - CARD_W - 12;
    if (left < 12) left = 12;
    card.style.left = `${left}px`;
    card.style.top = `${top}px`;
  }

  function render_passive() {
    if (root) render();
  }

  function render() {
    const step = STEPS[idx];
    const spot = root.querySelector(".tut-spot");
    const card = root.querySelector(".tut-card");
    root.querySelector(".tut-title").textContent = step.title;
    root.querySelector(".tut-body").textContent = step.body;
    root.querySelector(".tut-back").style.visibility = idx === 0 ? "hidden" : "visible";
    root.querySelector(".tut-next").textContent = idx === STEPS.length - 1 ? "done" : "next";
    root.querySelectorAll(".tut-dot").forEach((d, i) => d.classList.toggle("on", i === idx));

    const tgt = find_target(step);
    if (tgt) {
      root.classList.remove("tut-center");
      tgt.scrollIntoView({ block: "center", behavior: "auto" });
      const r = tgt.getBoundingClientRect();
      spot.style.left = `${r.left - PAD}px`;
      spot.style.top = `${r.top - PAD}px`;
      spot.style.width = `${r.width + PAD * 2}px`;
      spot.style.height = `${r.height + PAD * 2}px`;
      place_card(card, r);
    } else {
      root.classList.add("tut-center");
      place_card(card, null);
    }
  }

  function go(next) {
    if (next < 0) return;
    if (next >= STEPS.length) return finish();
    idx = next;
    switch_tab(STEPS[idx].tab);
    setTimeout(render, 80);
  }

  function finish() {
    post_settings({ tutorial_completed: true, tutorial_enabled: false });
    window.removeEventListener("resize", render_passive);
    if (root) { root.remove(); root = null; }
  }

  function start(from) {
    if (root) return;
    idx = typeof from === "number" ? from : 0;
    root = build_root();
    switch_tab(STEPS[idx].tab);
    setTimeout(render, 80);
  }

  function restart() {
    if (root) { root.remove(); root = null; }
    post_settings({ tutorial_completed: false }).then(() => start(0));
  }

  function auto_start() {
    get_settings().then(s => {
      const enabled = s.tutorial_enabled !== false;
      if (enabled && !s.tutorial_completed) start(0);
    });
  }

  window.Tutorial = { start, restart, finish, set_enabled, STEPS };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", auto_start);
  } else {
    auto_start();
  }
})();
