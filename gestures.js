// Rule-based Bharatanatyam hasta (single-hand mudra) classifier.
// Input: 21 MediaPipe hand landmarks [{x,y,z}]. Rotation invariant: uses joint angles
// and distances normalised by palm size, so hand orientation and distance don't matter.
const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
const d3 = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
function angle(a, b, c) {
  const u = [a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0)];
  const v = [c.x - b.x, c.y - b.y, (c.z || 0) - (b.z || 0)];
  const dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  const m = Math.hypot(...u) * Math.hypot(...v) || 1;
  return (Math.acos(clamp(dot / m, -1, 1)) * 180) / Math.PI;
}
const FINGERS = { I: [5, 6, 7, 8], M: [9, 10, 11, 12], R: [13, 14, 15, 16], P: [17, 18, 19, 20] };

export const MUDRAS = [
  { id: 'pataka', name: 'Pataka', meaning: 'Flag', how: 'Four fingers straight and together, thumb folded in.', spec: { T: 'in', I: 'ext', M: 'ext', R: 'ext', P: 'ext' } },
  { id: 'tripataka', name: 'Tripataka', meaning: 'Three-part flag', how: 'Like Pataka, but bend the ring finger.', spec: { T: 'in', I: 'ext', M: 'ext', R: 'bent', P: 'ext' } },
  { id: 'ardhapataka', name: 'Ardhapataka', meaning: 'Half flag', how: 'Index and middle straight and together; ring and little fingers bent.', spec: { T: 'in', I: 'ext', M: 'ext', R: 'bent', P: 'bent' } },
  { id: 'kartarimukha', name: 'Kartarimukha', meaning: 'Scissor face', how: 'Index and middle straight and spread apart; other two bent.', spec: { T: 'in', I: 'ext', M: 'ext', R: 'bent', P: 'bent', spread: true } },
  { id: 'mayura', name: 'Mayura', meaning: 'Peacock', how: 'Thumb tip touches the ring fingertip; other three fingers straight.', spec: { T: 'touchR', I: 'ext', M: 'ext', R: 'bent', P: 'ext' } },
  { id: 'hamsasya', name: 'Hamsasya', meaning: 'Swan beak', how: 'Thumb tip touches the index fingertip; other three fingers straight.', spec: { T: 'touchI', I: 'bent', M: 'ext', R: 'ext', P: 'ext' } },
  { id: 'suchi', name: 'Suchi', meaning: 'Needle', how: 'Index finger straight up, other fingers and thumb folded.', spec: { T: 'in', I: 'ext', M: 'bent', R: 'bent', P: 'bent' } },
  { id: 'chandrakala', name: 'Chandrakala', meaning: 'Moon crescent', how: 'Index and thumb both stretched out (an L), other fingers folded.', spec: { T: 'out', I: 'ext', M: 'bent', R: 'bent', P: 'bent' } },
  { id: 'mushti', name: 'Mushti', meaning: 'Fist', how: 'All fingers curled into a fist, thumb over the fingers.', spec: { T: 'in', I: 'bent', M: 'bent', R: 'bent', P: 'bent' } },
  { id: 'shikhara', name: 'Shikhara', meaning: 'Peak', how: 'Fist with the thumb pointing straight up.', spec: { T: 'out', I: 'bent', M: 'bent', R: 'bent', P: 'bent' } },
];

export function features(lm) {
  const scale = d3(lm[0], lm[9]) || 1e-6; // wrist to middle-finger knuckle
  const f = {};
  for (const [k, [m, p, dd, t]] of Object.entries(FINGERS)) {
    const ang = angle(lm[m], lm[p], lm[dd]);
    const ang2 = angle(lm[p], lm[dd], lm[t]);
    const reach = d3(lm[t], lm[0]) / d3(lm[m], lm[0]);
    // extended score: straight joints and tip well beyond the knuckle
    f[k] = clamp(((ang + ang2) / 2 - 115) / 45) * 0.6 + clamp((reach - 1.0) / 0.55) * 0.4;
  }
  const nearest = Math.min(...[5, 6, 9, 10, 11, 13, 14, 15, 17, 18].map((i) => d3(lm[4], lm[i])));
  const thumbRaw = clamp((nearest / scale - 0.4) / 0.4) * 0.5 + clamp((angle(lm[2], lm[3], lm[4]) - 135) / 35) * 0.5;
  const thumbOut = clamp(0.5 + (thumbRaw - 0.5) * 1.5);
  const pinch = (a, b) => clamp((0.65 - d3(lm[a], lm[b]) / scale) / 0.4);
  const gap = d3(lm[8], lm[12]) / scale;
  return { f, thumbOut, pinchI: pinch(4, 8), pinchR: pinch(4, 16), pinchM: pinch(4, 12), spread: clamp((gap - 0.28) / 0.3), together: clamp((0.5 - gap) / 0.25) };
}

export function scoreAll(lm) {
  const x = features(lm);
  const ext = (k) => x.f[k];
  const bent = (k) => 1 - x.f[k];
  const out = {};
  for (const m of MUDRAS) {
    const s = m.spec;
    const parts = [];
    for (const k of ['I', 'M', 'R', 'P']) {
      if (s.T === 'touchI' && k === 'I') continue; // index curls to meet thumb
      if (s.T === 'touchR' && k === 'R') continue;
      parts.push(s[k] === 'ext' ? ext(k) : bent(k));
    }
    if (s.T === 'in') parts.push(1 - x.thumbOut);
    if (m.id === 'tripataka') parts.push(1 - x.pinchR);
    if (s.T === 'out') parts.push(x.thumbOut);
    if (s.T === 'touchI') parts.push(x.pinchI);
    if (s.T === 'touchR') parts.push(x.pinchR);
    if (s.spread) parts.push(x.spread);
    else if (s.I === 'ext' && s.M === 'ext') parts.push(x.together);
    // soft-min: the weakest condition dominates, so one wrong finger rejects the pose
    const min = Math.min(...parts);
    const mean = parts.reduce((a, b) => a + b, 0) / parts.length;
    out[m.id] = 0.7 * min + 0.3 * mean;
  }
  return { scores: out, features: x };
}

export function classify(lm, threshold = 0.42) {
  const { scores, features: fe } = scoreAll(lm);
  const ranked = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const [id, sc] = ranked[0];
  const margin = sc - ranked[1][1];
  // Thumb-only pairs (same fingers, thumb in vs out): on a near tie, the thumb decides; an undecided thumb counts as folded in.
  const PAIRS = [['suchi', 'chandrakala'], ['mushti', 'shikhara']];
  let pick = id;
  if (margin < 0.03) {
    const pair = PAIRS.find((q) => q.includes(id) && q.includes(ranked[1][0]));
    if (pair) pick = fe.thumbOut > 0.55 ? pair[1] : pair[0];
  }
  const tie = margin < 0.03 && pick === id ? !PAIRS.some((q) => q.includes(id) && q.includes(ranked[1][0])) : false;
  const ok = sc >= threshold && !tie;
  return { id: ok ? pick : null, best: id, score: sc, margin, scores, features: fe };
}
