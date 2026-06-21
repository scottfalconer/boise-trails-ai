const CACHE_NAME = "boise-trails-field-packet-v18-983c773fdb827ac7a0";
const PRECACHE_URLS = [
  "./",
  "index.html",
  "manifest.json",
  "manifest.webmanifest",
  "icons/icon-192.png",
  "icons/icon-512.png",
  "gpx/all-field-packet-gpx.zip",
  "field-tool-data.json",
  "live-map.html",
  "gpx/official/upper-hulls-gulch-scott-s-4b.gpx",
  "gpx/cues/upper-hulls-gulch-scott-s-4b.gpx",
  "gpx/audit/upper-hulls-gulch-scott-s-4b.gpx",
  "gpx/official/western-foothills-wild-phlox-7.gpx",
  "gpx/cues/western-foothills-wild-phlox-7.gpx",
  "gpx/audit/western-foothills-wild-phlox-7.gpx",
  "gpx/official/dry-creek-barn-owl-5a.gpx",
  "gpx/cues/dry-creek-barn-owl-5a.gpx",
  "gpx/audit/dry-creek-barn-owl-5a.gpx",
  "gpx/official/hillside-to-hollow-full-sail-1a-2.gpx",
  "gpx/cues/hillside-to-hollow-full-sail-1a-2.gpx",
  "gpx/audit/hillside-to-hollow-full-sail-1a-2.gpx",
  "gpx/official/upper-hulls-gulch-bob-s-4a.gpx",
  "gpx/cues/upper-hulls-gulch-bob-s-4a.gpx",
  "gpx/audit/upper-hulls-gulch-bob-s-4a.gpx",
  "gpx/official/13-36th-street-chute-1a-1.gpx",
  "gpx/cues/13-36th-street-chute-1a-1.gpx",
  "gpx/audit/13-36th-street-chute-1a-1.gpx",
  "gpx/official/boise-river-wma-harris-ridge-trail-8.gpx",
  "gpx/cues/boise-river-wma-harris-ridge-trail-8.gpx",
  "gpx/audit/boise-river-wma-harris-ridge-trail-8.gpx",
  "gpx/official/hidden-springs-red-tail-15a.gpx",
  "gpx/cues/hidden-springs-red-tail-15a.gpx",
  "gpx/audit/hidden-springs-red-tail-15a.gpx",
  "gpx/official/hawkins-range-reserve-hawkins-11.gpx",
  "gpx/cues/hawkins-range-reserve-hawkins-11.gpx",
  "gpx/audit/hawkins-range-reserve-hawkins-11.gpx",
  "gpx/official/hidden-springs-bitterbrush-10b.gpx",
  "gpx/cues/hidden-springs-bitterbrush-10b.gpx",
  "gpx/audit/hidden-springs-bitterbrush-10b.gpx",
  "gpx/official/western-foothills-veterans-9.gpx",
  "gpx/cues/western-foothills-veterans-9.gpx",
  "gpx/audit/western-foothills-veterans-9.gpx",
  "gpx/official/cervidae-arrow-rock-cervidae-peak-19.gpx",
  "gpx/cues/cervidae-arrow-rock-cervidae-peak-19.gpx",
  "gpx/audit/cervidae-arrow-rock-cervidae-peak-19.gpx",
  "gpx/official/polecat-gulch-polecat-loop-5b.gpx",
  "gpx/cues/polecat-gulch-polecat-loop-5b.gpx",
  "gpx/audit/polecat-gulch-polecat-loop-5b.gpx",
  "gpx/official/table-rock-shoshone-paiute-tribes-trail-4c.gpx",
  "gpx/cues/table-rock-shoshone-paiute-tribes-trail-4c.gpx",
  "gpx/audit/table-rock-shoshone-paiute-tribes-trail-4c.gpx",
  "gpx/official/rocky-canyon-orchard-gulch-14.gpx",
  "gpx/cues/rocky-canyon-orchard-gulch-14.gpx",
  "gpx/audit/rocky-canyon-orchard-gulch-14.gpx",
  "gpx/official/camels-back-hulls-gulch-lower-hull-s-gulch-2.gpx",
  "gpx/cues/camels-back-hulls-gulch-lower-hull-s-gulch-2.gpx",
  "gpx/audit/camels-back-hulls-gulch-lower-hull-s-gulch-2.gpx",
  "gpx/official/dry-creek-16a-d1.gpx",
  "gpx/cues/dry-creek-16a-d1.gpx",
  "gpx/audit/dry-creek-16a-d1.gpx",
  "gpx/official/bogus-basin-sunshine-xc-17.gpx",
  "gpx/cues/bogus-basin-sunshine-xc-17.gpx",
  "gpx/audit/bogus-basin-sunshine-xc-17.gpx"
];
const NETWORK_FIRST_URLS = new Set([
  "index.html",
  "live-map.html",
  "field-tool-data.json",
  "manifest.json",
  "manifest.webmanifest"
]);

function normalizedCacheKey(request) {
  const requestUrl = new URL(request.url);
  requestUrl.search = '';
  return requestUrl.href;
}

function shouldUseNetworkFirst(request) {
  const requestUrl = new URL(request.url);
  const filename = requestUrl.pathname.split('/').pop() || 'index.html';
  return NETWORK_FIRST_URLS.has(filename) || requestUrl.pathname.includes('/gpx/');
}

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys => Promise.all(
      keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key))
    )).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', event => {
  if (event.request.method !== 'GET') {
    return;
  }
  const cacheKey = normalizedCacheKey(event.request);
  if (shouldUseNetworkFirst(event.request)) {
    event.respondWith(
      fetch(event.request).then(response => {
        const copy = response.clone();
        caches.open(CACHE_NAME).then(cache => cache.put(cacheKey, copy));
        return response;
      }).catch(() => caches.match(cacheKey).then(cached => cached || caches.match('./index.html')))
    );
    return;
  }
  event.respondWith(
    caches.match(event.request).then(cached => {
      if (cached) {
        return cached;
      }
      return fetch(event.request).then(response => {
        const copy = response.clone();
        caches.open(CACHE_NAME).then(cache => cache.put(event.request, copy));
        return response;
      }).catch(() => caches.match('./index.html'));
    })
  );
});
