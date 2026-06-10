const CACHE_NAME = "boise-trails-field-packet-v31-3897f2dc04cca69081";
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
  "gpx/official/boise-river-wma-peace-valley-overlook-8b.gpx",
  "gpx/cues/boise-river-wma-peace-valley-overlook-8b.gpx",
  "gpx/audit/boise-river-wma-peace-valley-overlook-8b.gpx",
  "gpx/official/dry-creek-sheep-camp-16a-2.gpx",
  "gpx/cues/dry-creek-sheep-camp-16a-2.gpx",
  "gpx/audit/dry-creek-sheep-camp-16a-2.gpx",
  "gpx/official/hillside-to-hollow-full-sail-1a-2.gpx",
  "gpx/cues/hillside-to-hollow-full-sail-1a-2.gpx",
  "gpx/audit/hillside-to-hollow-full-sail-1a-2.gpx",
  "gpx/official/upper-hulls-gulch-bob-s-4a.gpx",
  "gpx/cues/upper-hulls-gulch-bob-s-4a.gpx",
  "gpx/audit/upper-hulls-gulch-bob-s-4a.gpx",
  "gpx/official/boise-river-wma-harris-ridge-trail-8a.gpx",
  "gpx/cues/boise-river-wma-harris-ridge-trail-8a.gpx",
  "gpx/audit/boise-river-wma-harris-ridge-trail-8a.gpx",
  "gpx/official/13-36th-street-chute-1a-1.gpx",
  "gpx/cues/13-36th-street-chute-1a-1.gpx",
  "gpx/audit/13-36th-street-chute-1a-1.gpx",
  "gpx/official/hillside-to-hollow-who-now-loop-1b.gpx",
  "gpx/cues/hillside-to-hollow-who-now-loop-1b.gpx",
  "gpx/audit/hillside-to-hollow-who-now-loop-1b.gpx",
  "gpx/official/hidden-springs-red-tail-15a.gpx",
  "gpx/cues/hidden-springs-red-tail-15a.gpx",
  "gpx/audit/hidden-springs-red-tail-15a.gpx",
  "gpx/official/hawkins-range-reserve-hawkins-11.gpx",
  "gpx/cues/hawkins-range-reserve-hawkins-11.gpx",
  "gpx/audit/hawkins-range-reserve-hawkins-11.gpx",
  "gpx/official/hidden-springs-bitterbrush-10b.gpx",
  "gpx/cues/hidden-springs-bitterbrush-10b.gpx",
  "gpx/audit/hidden-springs-bitterbrush-10b.gpx",
  "gpx/official/bogus-basin-shindig-18b.gpx",
  "gpx/cues/bogus-basin-shindig-18b.gpx",
  "gpx/audit/bogus-basin-shindig-18b.gpx",
  "gpx/official/bogus-basin-stack-rock-connector-16c-1.gpx",
  "gpx/cues/bogus-basin-stack-rock-connector-16c-1.gpx",
  "gpx/audit/bogus-basin-stack-rock-connector-16c-1.gpx",
  "gpx/official/western-foothills-veterans-9.gpx",
  "gpx/cues/western-foothills-veterans-9.gpx",
  "gpx/audit/western-foothills-veterans-9.gpx",
  "gpx/official/cervidae-arrow-rock-cervidae-peak-19.gpx",
  "gpx/cues/cervidae-arrow-rock-cervidae-peak-19.gpx",
  "gpx/audit/cervidae-arrow-rock-cervidae-peak-19.gpx",
  "gpx/official/polecat-gulch-polecat-loop-5b.gpx",
  "gpx/cues/polecat-gulch-polecat-loop-5b.gpx",
  "gpx/audit/polecat-gulch-polecat-loop-5b.gpx",
  "gpx/official/military-reserve-military-reserve-connection-3.gpx",
  "gpx/cues/military-reserve-military-reserve-connection-3.gpx",
  "gpx/audit/military-reserve-military-reserve-connection-3.gpx",
  "gpx/official/dry-creek-corrals-12.gpx",
  "gpx/cues/dry-creek-corrals-12.gpx",
  "gpx/audit/dry-creek-corrals-12.gpx",
  "gpx/official/dry-creek-sweet-connie-16a-1.gpx",
  "gpx/cues/dry-creek-sweet-connie-16a-1.gpx",
  "gpx/audit/dry-creek-sweet-connie-16a-1.gpx",
  "gpx/official/table-rock-shoshone-paiute-tribes-trail-4c.gpx",
  "gpx/cues/table-rock-shoshone-paiute-tribes-trail-4c.gpx",
  "gpx/audit/table-rock-shoshone-paiute-tribes-trail-4c.gpx",
  "gpx/official/rocky-canyon-orchard-gulch-14.gpx",
  "gpx/cues/rocky-canyon-orchard-gulch-14.gpx",
  "gpx/audit/rocky-canyon-orchard-gulch-14.gpx",
  "gpx/official/dry-creek-shingle-creek-16c-2.gpx",
  "gpx/cues/dry-creek-shingle-creek-16c-2.gpx",
  "gpx/audit/dry-creek-shingle-creek-16c-2.gpx",
  "gpx/official/avimor-harlow-twisted-spring-h1.gpx",
  "gpx/cues/avimor-harlow-twisted-spring-h1.gpx",
  "gpx/audit/avimor-harlow-twisted-spring-h1.gpx",
  "gpx/official/camels-back-hulls-gulch-lower-hull-s-gulch-2.gpx",
  "gpx/cues/camels-back-hulls-gulch-lower-hull-s-gulch-2.gpx",
  "gpx/audit/camels-back-hulls-gulch-lower-hull-s-gulch-2.gpx",
  "gpx/official/bogus-basin-brewers-byway-ext-18a.gpx",
  "gpx/cues/bogus-basin-brewers-byway-ext-18a.gpx",
  "gpx/audit/bogus-basin-brewers-byway-ext-18a.gpx",
  "gpx/official/dry-creek-highlands-15b.gpx",
  "gpx/cues/dry-creek-highlands-15b.gpx",
  "gpx/audit/dry-creek-highlands-15b.gpx",
  "gpx/official/bogus-basin-sunshine-xc-17.gpx",
  "gpx/cues/bogus-basin-sunshine-xc-17.gpx",
  "gpx/audit/bogus-basin-sunshine-xc-17.gpx",
  "gpx/official/dry-creek-peggy-s-6.gpx",
  "gpx/cues/dry-creek-peggy-s-6.gpx",
  "gpx/audit/dry-creek-peggy-s-6.gpx",
  "gpx/official/rocky-canyon-three-bears-13.gpx",
  "gpx/cues/rocky-canyon-three-bears-13.gpx",
  "gpx/audit/rocky-canyon-three-bears-13.gpx"
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
