// The side cards use only the existing fixed-view images. The 3D viewer in
// the middle remains the sole WebGPU renderer.
export function initCarousel(subjects, selected, getFrame) {
  const stage = document.querySelector('.carousel-stage');
  const carousel = document.querySelector('.carousel');
  const nav = document.querySelector('.carousel-nav');
  const title = document.getElementById('subject-title');
  const position = subjects.findIndex(subject => subject.id === selected.id);
  const subjectAt = offset => subjects[(position + offset + subjects.length) % subjects.length];
  const cards = [
    { element: document.getElementById('previous-card'), subject: subjectAt(-1), offset: -1 },
    { element: document.getElementById('next-card'), subject: subjectAt(1), offset: 1 }
  ];
  let changing = false;
  let suppressClick = false;
  title.textContent = `Subject ${selected.id}`;

  function switchSubject(offset) {
    if (changing || subjects.length < 2) return;
    changing = true;
    stage.classList.add(offset > 0 ? 'leaving-left' : 'leaving-right');
    const url = new URL(location.href);
    url.searchParams.set('subject', subjectAt(offset).id);
    const delay = matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 200;
    setTimeout(() => location.assign(url), delay);
  }
  const previous = document.getElementById('previous-subject');
  const next = document.getElementById('next-subject');
  previous.disabled = next.disabled = subjects.length < 2;
  previous.addEventListener('click', () => switchSubject(-1));
  next.addEventListener('click', () => switchSubject(1));
  for (const card of cards) {
    card.element.disabled = subjects.length < 2;
    card.element.setAttribute('aria-label', `View subject ${card.subject.id}`);
    card.element.querySelector('.side-label').textContent = `Subject ${card.subject.id}`;
    card.element.addEventListener('click', event => {
      if (suppressClick) { event.preventDefault(); return; }
      switchSubject(card.offset);
    });
  }

  // Dragging either side card or the navigation switches subjects. Pointer
  // events inside the central canvas are reserved for camera manipulation.
  let swipe = null;
  for (const area of [...cards.map(card => card.element), nav]) {
    area.addEventListener('pointerdown', event => {
      if (event.pointerType === 'mouse' && event.button !== 0) return;
      swipe = { x: event.clientX, y: event.clientY, id: event.pointerId };
      // Preserve the clicked button as the pointer target. Capturing on the
      // navigation container would suppress its child buttons' click events.
      event.target.setPointerCapture(event.pointerId);
    });
    area.addEventListener('pointerup', event => {
      if (!swipe || swipe.id !== event.pointerId) return;
      const dx = event.clientX - swipe.x, dy = event.clientY - swipe.y;
      swipe = null;
      if (Math.abs(dx) > 35 && Math.abs(dx) > Math.abs(dy) * 1.2) {
        suppressClick = true;
        switchSubject(dx < 0 ? 1 : -1);
        setTimeout(() => { suppressClick = false; }, 400);
      }
    });
    area.addEventListener('pointercancel', () => { swipe = null; });
  }

  let visible = false;
  const observer = new IntersectionObserver(entries => {
    visible = entries.some(entry => entry.isIntersecting);
  }, { rootMargin: '100px' });
  observer.observe(carousel);
  let started = false;
  async function setup(card) {
    const base = card.subject.id === '00016'
      ? new URL('../data/', import.meta.url)
      : new URL(`../data/subjects/${card.subject.id}/`, import.meta.url);
    const status = card.element.querySelector('.side-status');
    try {
      const response = await fetch(new URL('metadata.json', base));
      if (!response.ok) throw new Error('Preview metadata is unavailable.');
      const metadata = await response.json();
      const info = metadata.preview;
      if (!info || typeof info.path !== 'string' || !Number.isInteger(info.count) || info.count < 1) throw new Error('Preview metadata is invalid.');
      Object.assign(card, { base, info, shown: -1, loading: false });
    } catch (error) {
      status.textContent = 'Preview unavailable';
      card.unavailable = true;
      console.warn(`Could not load subject ${card.subject.id} preview.`, error);
    }
  }
  function updateCard(card, frame) {
    if (!card.info || card.loading || card.unavailable) return;
    const index = frame.count ? Math.min(card.info.count - 1, Math.floor(frame.index * card.info.count / frame.count)) : 0;
    if (index === card.shown) return;
    // At most one request per card is in flight, and no array of decoded
    // frames is retained; the browser may cache the response normally.
    card.loading = true;
    const image = new Image();
    image.onload = () => {
      card.element.querySelector('img').src = image.src;
      card.element.querySelector('.side-status').hidden = true;
      card.shown = index;
      card.loading = false;
    };
    image.onerror = () => {
      card.loading = false;
      card.unavailable = true;
      const status = card.element.querySelector('.side-status');
      status.textContent = 'Preview unavailable';
      status.hidden = false;
    };
    image.src = new URL(`${card.info.path}${String(index).padStart(6, '0')}.${card.info.format || 'jpg'}`, card.base).href;
  }
  function tick() {
    if (visible) {
      if (!started) { started = true; cards.forEach(card => void setup(card)); }
      const frame = getFrame();
      for (const card of cards) updateCard(card, frame);
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
