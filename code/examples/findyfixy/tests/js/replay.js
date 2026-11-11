// Replays a recorded event timeline through the page's own animation code at REAL
// speed and prints, as JSON, when each event arrived and when the picture began to
// show it. Used by tests/test_page_lag.py.
// Usage: node replay.js page.html events.json [seconds to wait after the last event] [ids] [motion]
// With `ids` (comma-separated element ids), it prints {"rows": [...], "text": {id: text}}
// instead, so a test can read what the page ended up showing. `text` is an element's
// textContent, or its children's when it has any (the result panel is a list).
// `motion` "reduced" answers the page's prefers-reduced-motion query with yes: the page's
// own no-animation path, so a long stream plays in moments whatever its data says.
// Used by tests/test_page_truth.py.
//
// The DOM is a stub: only what the animation touches. Normal motion unless asked, so the
// real timings are the ones measured.
const fs = require("fs");
const html = fs.readFileSync(process.argv[2], "utf8");
const events = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const js = html.match(/<script>([\s\S]*)<\/script>/)[1].replace('"use strict";', "");

const els = {};
function el(id) {
  if (!els[id]) {
    const cls = new Set();
    els[id] = {
      id, value: "", hidden: false, textContent: "", children: [], style: {}, scrollTop: 0,
      scrollHeight: 0, className: "", disabled: false,
      classList: { add: (...c) => c.forEach((x) => cls.add(x)), remove: (...c) => c.forEach((x) => cls.delete(x)), toggle() {} },
      setAttribute() {}, getTotalLength: () => 100, getPointAtLength: () => ({ x: 0, y: 0 }),
      addEventListener() {}, append(...a) { this.children.push(...a); },
      replaceChildren() { this.children = []; }, querySelector: () => ({ disabled: false }),
    };
  }
  return els[id];
}
global.document = { getElementById: el, createElement: () => el("x" + Math.random()), querySelectorAll: () => [], hidden: false };
const reduced = process.argv[6] === "reduced";
global.window = { matchMedia: () => ({ matches: reduced }) };
global.performance = { now: () => Date.now() };
global.requestAnimationFrame = (f) => setTimeout(() => f(Date.now()), 16);
global.fetch = async () => ({ json: async () => ({ scenarios: [], questions: [], incidents: [], live_available: true }) });
let es;
global.EventSource = function () { es = this; this.close = () => {}; };
global.Option = function () { return {}; };
global.setInterval = () => 1;
global.location = { reload() {} };

const test = `
const t0 = Date.now();
const rows = [];
const orig = animate;
animate = async (ev) => {
  const started = (Date.now() - t0) / 1000;
  await orig(ev);
  rows.push({ kind: ev.kind, node: ev.data.node || ev.data.model || "", real: ev.t, started });
};
running = true;
openStream();
for (const ev of ${JSON.stringify(events)}) setTimeout(() => es.onmessage({ data: JSON.stringify(ev) }), ev.t * 1000);
const last = ${JSON.stringify(events)}.at(-1).t;
const ids = ${JSON.stringify((process.argv[5] || "").split(",").filter(Boolean))};
const shown = (e) => e.children.length ? e.children.map((c) => c.textContent) : e.textContent;
const report = () => ids.length ? { rows, text: Object.fromEntries(ids.map((id) => [id, shown(document.getElementById(id))])) } : rows;
setTimeout(() => { console.log(JSON.stringify(report())); process.exit(0); }, (last + ${Number(process.argv[4] || 4)}) * 1000);
`;
eval(js + test);
