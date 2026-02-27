const catalogGrid = document.getElementById('catalogGrid');
const recommendedGrid = document.getElementById('recommendedGrid');
const detailSection = document.getElementById('detailSection');
const detailTitle = document.getElementById('detailTitle');
const detailGenre = document.getElementById('detailGenre');
const detailSynopsis = document.getElementById('detailSynopsis');
const searchInput = document.getElementById('searchInput');

async function apiGet(url) {
  const resp = await fetch(url);
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.detail || 'Request failed');
  return data;
}

function synopsisPreview(text, maxLen = 150) {
  if (!text) return 'No synopsis available.';
  return text.length > maxLen ? `${text.slice(0, maxLen)}...` : text;
}

function renderAnimeCard(anime, onClick) {
  const card = document.createElement('article');
  card.className = 'card';
  card.innerHTML = `
    <h4>${anime.title}</h4>
    <p><strong>Genre:</strong> ${anime.genre || '-'}</p>
    <p>${synopsisPreview(anime.synopsis)}</p>
  `;
  card.addEventListener('click', () => onClick(anime.anime_id));
  return card;
}

async function loadCatalog(query = '') {
  catalogGrid.innerHTML = '<p>Loading catalog...</p>';
  try {
    const endpoint = query ? `/anime?limit=80&q=${encodeURIComponent(query)}` : '/anime?limit=80';
    const data = await apiGet(endpoint);
    catalogGrid.innerHTML = '';
    data.anime.forEach((item) => {
      catalogGrid.appendChild(renderAnimeCard(item, openAnimeDetail));
    });
  } catch (err) {
    catalogGrid.innerHTML = `<p>Error: ${err.message}</p>`;
  }
}

async function openAnimeDetail(animeId) {
  detailSection.classList.remove('hidden');
  recommendedGrid.innerHTML = '<p>Loading recommendations...</p>';

  try {
    const anime = await apiGet(`/anime/${animeId}`);
    detailTitle.textContent = anime.title;
    detailGenre.textContent = anime.genre || '-';
    detailSynopsis.textContent = anime.synopsis || 'No synopsis available.';

    const rec = await apiGet(`/recommend/anime/${animeId}`);
    recommendedGrid.innerHTML = '';

    for (const item of rec.recommendations) {
      recommendedGrid.appendChild(renderAnimeCard(item, openAnimeDetail));
    }

    history.pushState({ animeId }, '', `/anime/${animeId}`);
    window.scrollTo({ top: detailSection.offsetTop - 10, behavior: 'smooth' });
  } catch (err) {
    recommendedGrid.innerHTML = `<p>Error: ${err.message}</p>`;
  }
}

searchInput.addEventListener('input', () => {
  loadCatalog(searchInput.value.trim());
});

window.addEventListener('popstate', async () => {
  const match = window.location.pathname.match(/^\/anime\/(\d+)$/);
  if (match) {
    await openAnimeDetail(Number(match[1]));
  } else {
    detailSection.classList.add('hidden');
  }
});

(async function init() {
  await loadCatalog();
  const match = window.location.pathname.match(/^\/anime\/(\d+)$/);
  if (match) {
    await openAnimeDetail(Number(match[1]));
  }
})();
