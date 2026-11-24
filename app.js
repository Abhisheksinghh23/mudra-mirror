import { FilesetResolver, HandLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';
import { MUDRAS, classify } from './gestures.js';

const $ = (id) => document.getElementById(id);
const video = $('video'), canvas = $('overlay'), ctx = canvas.getContext('2d');
const BONES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const TEST = new URLSearchParams(location.search).has('test'); // test mode: Suchi first, 2.5 s rounds
const HOLD_MS = 800, ROUND_MS = TEST ? 2500 : 8000, ROUNDS = 8;

let landmarker = null, stream = null, running = false, lastVideoTime = -1;
let mode = 'learn', selected = null, liveId = null;
const hist = [];
let game = null;

/* ---------- schematic icons (drawn from each hasta's finger spec) ---------- */
function icon(m, size) {
  const s = m.spec, X = { I: 38, M: 54, R: 70, P: 86 };
  let g = `<rect x="30" y="82" width="68" height="58" rx="14" fill="#5a2a31" stroke="#e8b04a" stroke-width="2"/>`;
  for (const k of ['I', 'M', 'R', 'P']) {
    const ext = s[k] === 'ext';
    const rot = s.spread && k === 'I' ? -14 : s.spread && k === 'M' ? 10 : 0;
    const h = ext ? 62 : 20;
    const col = ext ? '#f6ead8' : '#9a7468';
    g += `<g transform="rotate(${rot} ${X[k]} 84)"><rect x="${X[k] - 6}" y="${84 - h}" width="12" height="${h}" rx="6" fill="${col}"/></g>`;
  }
  const tx = 30, ty = 112;
  if (s.T === 'out') g += `<line x1="${tx}" y1="${ty}" x2="6" y2="70" stroke="#f6ead8" stroke-width="12" stroke-linecap="round"/>`;
  else if (s.T === 'touchI') g += `<line x1="${tx}" y1="${ty}" x2="36" y2="70" stroke="#f6ead8" stroke-width="12" stroke-linecap="round"/><circle cx="38" cy="66" r="9" fill="none" stroke="#e8b04a" stroke-width="3"/>`;
  else if (s.T === 'touchR') g += `<line x1="${tx}" y1="${ty}" x2="66" y2="76" stroke="#f6ead8" stroke-width="12" stroke-linecap="round"/><circle cx="70" cy="72" r="9" fill="none" stroke="#e8b04a" stroke-width="3"/>`;
  else g += `<line x1="${tx}" y1="${ty}" x2="52" y2="118" stroke="#9a7468" stroke-width="12" stroke-linecap="round"/>`;
  return g;
}
function drawIcon(svg, m) { svg.innerHTML = icon(m); }

/* ---------- UI ---------- */
function pick(m){selected=m;$('dName').textContent=m.name;$('dMeaning').textContent=m.meaning;$('dHow').textContent=m.how;drawIcon($('detailIcon'),m);}
const grid = $('grid');
for (const m of MUDRAS) {
  const li = document.createElement('li');
  li.innerHTML = `<button class="card" type="button" data-id="${m.id}"><svg viewBox="0 0 120 150" width="64" height="80" aria-hidden="true">${icon(m)}</svg><b>${m.name}</b><span>${m.meaning}</span></button>`;
  grid.append(li);
}
grid.addEventListener('click', (e) => {
  const b = e.target.closest('.card'); if (!b) return;
  pick(MUDRAS.find((m) => m.id === b.dataset.id));
  refreshCards();
});
pick(MUDRAS[0]);
function refreshCards() {
  grid.querySelectorAll('.card').forEach((c) => {
    c.classList.toggle('sel', selected?.id === c.dataset.id);
    c.classList.toggle('live', liveId === c.dataset.id);
  });
}
function setMode(m) {
  mode = m;
  $('tabLearn').classList.toggle('on', m === 'learn'); $('tabGame').classList.toggle('on', m === 'game');
  $('learnPane').hidden = m !== 'learn'; $('gamePane').hidden = m !== 'game';
  if (m === 'learn') endGame(true);
  $('gameHud').hidden = true;
}
$('tabLearn').onclick = () => setMode('learn');
$('tabGame').onclick = () => setMode('game');
const bestKey = 'mudra-mirror-best';
function loadBest() { try { return +localStorage.getItem(bestKey) || 0; } catch { return 0; } }
function saveBest(v) { try { localStorage.setItem(bestKey, String(v)); } catch {} }
function showBest() { const b = loadBest(); $('best').textContent = b ? `Best score on this device: ${b}` : ''; }
showBest();

function setStatus(t, err = false) { const s = $('status'); s.textContent = t; s.classList.toggle('err', err); }

/* ---------- model + camera ---------- */
async function loadModel() {
  if (landmarker) return;
  setStatus('Loading hand model...');
  const fileset = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
  const mk = (delegate) => HandLandmarker.createFromOptions(fileset, { baseOptions: { modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', delegate }, runningMode: 'VIDEO', numHands: 1, minHandDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
  try { landmarker = await mk('GPU'); } catch { landmarker = await mk('CPU'); }
}
async function start() {
  $('startBtn').disabled = true;
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('This browser has no camera access here. Camera needs a secure (https) page. You can still try a photo.');
    await loadModel();
    setStatus('Waiting for camera permission...');
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
    video.srcObject = stream; await video.play();
    $('viewport').classList.remove('photo');
    $('splash').hidden = true; $('hud').hidden = false; $('stopBtn').hidden = false;
    running = true; lastVideoTime = -1; requestAnimationFrame(loop);
  } catch (e) {
    const msg = e.name === 'NotAllowedError' ? 'Camera permission was denied. Allow it in your browser settings, or try a photo.' : e.name === 'NotFoundError' ? 'No camera found. You can still try a photo.' : e.message || String(e);
    setStatus(msg, true); $('startBtn').disabled = false; $('startBtn').textContent = 'Try again';
  }
}
function stop() {
  running = false; stream?.getTracks().forEach((t) => t.stop()); stream = null; video.srcObject = null;
  $('splash').hidden = false; $('hud').hidden = true; $('stopBtn').hidden = true; $('startBtn').disabled = false; $('startBtn').textContent = 'Start camera';
  setStatus('Camera stopped.'); endGame(true); ctx.clearRect(0, 0, canvas.width, canvas.height); liveId = null; refreshCards();
}
$('startBtn').onclick = start; $('stopBtn').onclick = stop;

function draw(lm, w, h, color) {
  canvas.width = w; canvas.height = h; ctx.clearRect(0, 0, w, h);
  if (!lm) return;
  ctx.lineWidth = Math.max(2, w / 240); ctx.strokeStyle = color; ctx.fillStyle = '#fff';
  for (const [a, b] of BONES) { ctx.beginPath(); ctx.moveTo(lm[a].x * w, lm[a].y * h); ctx.lineTo(lm[b].x * w, lm[b].y * h); ctx.stroke(); }
  for (const p of lm) { ctx.beginPath(); ctx.arc(p.x * w, p.y * h, Math.max(2.5, w / 200), 0, 7); ctx.fill(); }
}

function stable(id) {
  hist.push(id); if (hist.length > 7) hist.shift();
  const c = {}; for (const h of hist) c[h] = (c[h] || 0) + 1;
  const [top, n] = Object.entries(c).sort((a, b) => b[1] - a[1])[0];
  return n >= 4 && top !== 'null' ? top : null;
}

function loop() {
  if (!running) return;
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const res = landmarker.detectForVideo(video, performance.now());
    const lm = res.landmarks?.[0];
    let id = null, r = null;
    if (lm) { r = classify(lm); id = r.id; window.__last = r; }
    const s = stable(lm ? id : null);
    if (!lm) hist.length = 0;
    draw(lm, video.videoWidth, video.videoHeight, s ? '#58c27d' : '#e8b04a');
    liveId = s; refreshCards();
    const hn = $('hudName');
    if (!lm) { hn.textContent = 'Show your hand'; hn.className = 'hud-name none'; $('hudConf').textContent = ''; }
    else if (s) { const m = MUDRAS.find((x) => x.id === s); hn.textContent = `${m.name} (${m.meaning})`; hn.className = 'hud-name'; $('hudConf').textContent = `Rule score ${Math.round(r.scores[s] * 100)}/100 (heuristic, not a confidence)`; }
    else { hn.textContent = 'Not sure'; hn.className = 'hud-name none'; $('hudConf').textContent = r ? `closest: ${MUDRAS.find((x) => x.id === r.best).name}, rule score ${Math.round(r.score * 100)}/100` : ''; }
    tickGame(s);
  }
  requestAnimationFrame(loop);
}

/* ---------- mirror game ---------- */
function shuffle(a) { a = a.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }
function startGame() {
  if (!running) { $('tHow').textContent = 'Start the camera first.'; return; }
  game = { order: TEST ? [MUDRAS.find((m) => m.id === 'suchi'), ...shuffle(MUDRAS.filter((m) => m.id !== 'suchi')).slice(0, ROUNDS - 1)] : shuffle(MUDRAS).slice(0, ROUNDS), i: 0, score: 0, roundStart: performance.now(), holdStart: null, got: 0 };
  $('gameHud').hidden = false; $('playBtn').hidden = true; showRound();
}
function showRound() {
  const m = game.order[game.i]; game.roundStart = performance.now(); game.holdStart = null;
  $('tKicker').textContent = `Round ${game.i + 1} of ${ROUNDS}`; $('tName').textContent = m.name; $('tHow').textContent = `${m.meaning}. ${m.how}`;
  drawIcon($('targetIcon'), m); $('ghRound').textContent = `Round ${game.i + 1}/${ROUNDS}`; $('ghScore').textContent = `${game.score} pts`;
}
function tickGame(s) {
  if (!game) return;
  const now = performance.now(), m = game.order[game.i];
  const left = Math.max(0, ROUND_MS - (now - game.roundStart));
  $('ghBar').style.transform = `scaleX(${left / ROUND_MS})`;
  if (s === m.id) {
    game.holdStart ??= now;
    if (now - game.holdStart >= HOLD_MS) { game.score += 100 + Math.round((left / 1000) * 10); game.got++; nextRound(true); return; }
  } else game.holdStart = null;
  if (left <= 0) nextRound(false);
}
function nextRound(hit) {
  game.i++; $('ghScore').textContent = `${game.score} pts`;
  if (game.i >= ROUNDS) return finishGame();
  showRound();
}
function finishGame() {
  const g = game, best = loadBest(); if (g.score > best) saveBest(g.score);
  $('tKicker').textContent = 'Done'; $('tName').textContent = `${g.score} points`;
  $('tHow').textContent = `You matched ${g.got} of ${ROUNDS}.${g.score > best ? ' New best!' : ''} If a correct hasta was not picked up, check light and hand angle: the camera reading, not you, may be the weak link.`;
  $('playBtn').hidden = false; $('playBtn').textContent = 'Play again'; $('gameHud').hidden = true; game = null; showBest();
}
function endGame(silent) { if (!game) return; game = null; $('gameHud').hidden = true; $('playBtn').hidden = false; $('tKicker').textContent = 'Mirror game'; $('tName').textContent = 'Ready?'; $('tHow').textContent = 'Eight rounds. Make the shown hasta and hold it for 0.8 seconds. Faster holds score more.'; }
$('playBtn').onclick = startGame;

/* ---------- photo test ---------- */
$('photo').addEventListener('change', async (e) => {
  const f = e.target.files[0]; if (!f) return;
  try {
    running = false; endGame(true);
    await loadModel();
    const img = new Image(); img.src = URL.createObjectURL(f); await img.decode();
    await landmarker.setOptions({ runningMode: 'IMAGE' });
    const res = landmarker.detect(img);
    await landmarker.setOptions({ runningMode: 'VIDEO' });
    const w = img.naturalWidth, h = img.naturalHeight;
    $('viewport').classList.add('photo'); $('splash').hidden = true; $('hud').hidden = false; $('stopBtn').hidden = false;
    canvas.width = w; canvas.height = h; ctx.drawImage(img, 0, 0);
    const lm = res.landmarks?.[0], hn = $('hudName');
    if (!lm) { hn.textContent = 'No hand found'; hn.className = 'hud-name none'; $('hudConf').textContent = ''; return; }
    const r = classify(lm);
    ctx.lineWidth = Math.max(3, w / 160); ctx.strokeStyle = '#e8b04a'; ctx.fillStyle = '#fff';
    for (const [a, b] of BONES) { ctx.beginPath(); ctx.moveTo(lm[a].x * w, lm[a].y * h); ctx.lineTo(lm[b].x * w, lm[b].y * h); ctx.stroke(); }
    for (const p of lm) { ctx.beginPath(); ctx.arc(p.x * w, p.y * h, Math.max(3, w / 140), 0, 7); ctx.fill(); }
    liveId = r.id; refreshCards();
    if (r.id) { const m = MUDRAS.find((x) => x.id === r.id); hn.textContent = `${m.name} (${m.meaning})`; hn.className = 'hud-name'; $('hudConf').textContent = `Rule score ${Math.round(r.score * 100)}/100 (heuristic, not a confidence)`; }
    else { hn.textContent = 'Not sure'; hn.className = 'hud-name none'; $('hudConf').textContent = `closest: ${MUDRAS.find((x) => x.id === r.best).name}`; }
    window.__lastPhoto = { id: r.id, best: r.best, score: r.score, f: r.features, s: r.scores };
  } catch (err) { setStatus(String(err.message || err), true); $('splash').hidden = false; }
  e.target.value = '';
});
$('stopBtn').addEventListener('click', () => $('viewport').classList.remove('photo'));
window.__mudra = { classify, MUDRAS };
