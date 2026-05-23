const CACHE_NAME = "boise-trails-field-packet-v45-0c035b533ac93f2ff5";
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
  "gpx/official/dry-creek-connector-fd28a.gpx",
  "gpx/cues/dry-creek-connector-fd28a.gpx",
  "gpx/audit/dry-creek-connector-fd28a.gpx",
  "gpx/official/camels-back-hulls-gulch-kestrel-fd19a.gpx",
  "gpx/cues/camels-back-hulls-gulch-kestrel-fd19a.gpx",
  "gpx/audit/camels-back-hulls-gulch-kestrel-fd19a.gpx",
  "gpx/official/polecat-gulch-doe-ridge-fd14a.gpx",
  "gpx/cues/polecat-gulch-doe-ridge-fd14a.gpx",
  "gpx/audit/polecat-gulch-doe-ridge-fd14a.gpx",
  "gpx/official/full-sail-n-36th-st-36th-street-chute-fd14c.gpx",
  "gpx/cues/full-sail-n-36th-st-36th-street-chute-fd14c.gpx",
  "gpx/audit/full-sail-n-36th-st-36th-street-chute-fd14c.gpx",
  "gpx/official/upper-hulls-gulch-scott-s-4b.gpx",
  "gpx/cues/upper-hulls-gulch-scott-s-4b.gpx",
  "gpx/audit/upper-hulls-gulch-scott-s-4b.gpx",
  "gpx/official/camels-back-hulls-gulch-owl-s-roost-fd22b.gpx",
  "gpx/cues/camels-back-hulls-gulch-owl-s-roost-fd22b.gpx",
  "gpx/audit/camels-back-hulls-gulch-owl-s-roost-fd22b.gpx",
  "gpx/official/hillside-to-hollow-who-now-loop-fd12b.gpx",
  "gpx/cues/hillside-to-hollow-who-now-loop-fd12b.gpx",
  "gpx/audit/hillside-to-hollow-who-now-loop-fd12b.gpx",
  "gpx/official/upper-hulls-gulch-bob-s-4a.gpx",
  "gpx/cues/upper-hulls-gulch-bob-s-4a.gpx",
  "gpx/audit/upper-hulls-gulch-bob-s-4a.gpx",
  "gpx/official/dry-creek-barn-owl-fd09a.gpx",
  "gpx/cues/dry-creek-barn-owl-fd09a.gpx",
  "gpx/audit/dry-creek-barn-owl-fd09a.gpx",
  "gpx/official/boise-river-wma-peace-valley-overlook-fd21b.gpx",
  "gpx/cues/boise-river-wma-peace-valley-overlook-fd21b.gpx",
  "gpx/audit/boise-river-wma-peace-valley-overlook-fd21b.gpx",
  "gpx/official/polecat-gulch-chbh-connector-fd14b.gpx",
  "gpx/cues/polecat-gulch-chbh-connector-fd14b.gpx",
  "gpx/audit/polecat-gulch-chbh-connector-fd14b.gpx",
  "gpx/official/camels-back-hulls-gulch-lower-hull-s-gulch-fd19b.gpx",
  "gpx/cues/camels-back-hulls-gulch-lower-hull-s-gulch-fd19b.gpx",
  "gpx/audit/camels-back-hulls-gulch-lower-hull-s-gulch-fd19b.gpx",
  "gpx/official/camels-back-hulls-gulch-crestline-fd22a.gpx",
  "gpx/cues/camels-back-hulls-gulch-crestline-fd22a.gpx",
  "gpx/audit/camels-back-hulls-gulch-crestline-fd22a.gpx",
  "gpx/official/dry-creek-sheep-camp-16a-2.gpx",
  "gpx/cues/dry-creek-sheep-camp-16a-2.gpx",
  "gpx/audit/dry-creek-sheep-camp-16a-2.gpx",
  "gpx/official/table-rock-fd21c.gpx",
  "gpx/cues/table-rock-fd21c.gpx",
  "gpx/audit/table-rock-fd21c.gpx",
  "gpx/official/hillside-to-hollow-bob-smylie-fd12a.gpx",
  "gpx/cues/hillside-to-hollow-bob-smylie-fd12a.gpx",
  "gpx/audit/hillside-to-hollow-bob-smylie-fd12a.gpx",
  "gpx/official/boise-river-wma-harris-ridge-trail-fd21a.gpx",
  "gpx/cues/boise-river-wma-harris-ridge-trail-fd21a.gpx",
  "gpx/audit/boise-river-wma-harris-ridge-trail-fd21a.gpx",
  "gpx/official/bogus-basin-sunshine-xc-fd07a.gpx",
  "gpx/cues/bogus-basin-sunshine-xc-fd07a.gpx",
  "gpx/audit/bogus-basin-sunshine-xc-fd07a.gpx",
  "gpx/official/dry-creek-cartwright-ridge-fd08a.gpx",
  "gpx/cues/dry-creek-cartwright-ridge-fd08a.gpx",
  "gpx/audit/dry-creek-cartwright-ridge-fd08a.gpx",
  "gpx/official/western-foothills-seaman-gulch-7.gpx",
  "gpx/cues/western-foothills-seaman-gulch-7.gpx",
  "gpx/audit/western-foothills-seaman-gulch-7.gpx",
  "gpx/official/cartwright-cartwright-connector-fd08b.gpx",
  "gpx/cues/cartwright-cartwright-connector-fd08b.gpx",
  "gpx/audit/cartwright-cartwright-connector-fd08b.gpx",
  "gpx/official/bogus-basin-stack-rock-connector-16b.gpx",
  "gpx/cues/bogus-basin-stack-rock-connector-16b.gpx",
  "gpx/audit/bogus-basin-stack-rock-connector-16b.gpx",
  "gpx/official/upper-hulls-gulch-hull-s-gulch-interpretive-trail-fd05a.gpx",
  "gpx/cues/upper-hulls-gulch-hull-s-gulch-interpretive-trail-fd05a.gpx",
  "gpx/audit/upper-hulls-gulch-hull-s-gulch-interpretive-trail-fd05a.gpx",
  "gpx/official/hidden-springs-red-tail-15b.gpx",
  "gpx/cues/hidden-springs-red-tail-15b.gpx",
  "gpx/audit/hidden-springs-red-tail-15b.gpx",
  "gpx/official/hawkins-range-reserve-hawkins-11.gpx",
  "gpx/cues/hawkins-range-reserve-hawkins-11.gpx",
  "gpx/audit/hawkins-range-reserve-hawkins-11.gpx",
  "gpx/official/hidden-springs-bitterbrush-10b.gpx",
  "gpx/cues/hidden-springs-bitterbrush-10b.gpx",
  "gpx/audit/hidden-springs-bitterbrush-10b.gpx",
  "gpx/official/dry-creek-chukar-butte-fd03a.gpx",
  "gpx/cues/dry-creek-chukar-butte-fd03a.gpx",
  "gpx/audit/dry-creek-chukar-butte-fd03a.gpx",
  "gpx/official/bogus-basin-deer-point-fd07b.gpx",
  "gpx/cues/bogus-basin-deer-point-fd07b.gpx",
  "gpx/audit/bogus-basin-deer-point-fd07b.gpx",
  "gpx/official/bogus-basin-the-face-fd25b.gpx",
  "gpx/cues/bogus-basin-the-face-fd25b.gpx",
  "gpx/audit/bogus-basin-the-face-fd25b.gpx",
  "gpx/official/table-rock-rock-island-east-fd01a.gpx",
  "gpx/cues/table-rock-rock-island-east-fd01a.gpx",
  "gpx/audit/table-rock-rock-island-east-fd01a.gpx",
  "gpx/official/bogus-basin-elk-meadows-fd25a.gpx",
  "gpx/cues/bogus-basin-elk-meadows-fd25a.gpx",
  "gpx/audit/bogus-basin-elk-meadows-fd25a.gpx",
  "gpx/official/western-foothills-veterans-9.gpx",
  "gpx/cues/western-foothills-veterans-9.gpx",
  "gpx/audit/western-foothills-veterans-9.gpx",
  "gpx/official/cervidae-arrow-rock-cervidae-peak-19.gpx",
  "gpx/cues/cervidae-arrow-rock-cervidae-peak-19.gpx",
  "gpx/audit/cervidae-arrow-rock-cervidae-peak-19.gpx",
  "gpx/official/military-reserve-two-point-fd04a.gpx",
  "gpx/cues/military-reserve-two-point-fd04a.gpx",
  "gpx/audit/military-reserve-two-point-fd04a.gpx",
  "gpx/official/upper-military-reserve-fat-tire-traverse-fd06a.gpx",
  "gpx/cues/upper-military-reserve-fat-tire-traverse-fd06a.gpx",
  "gpx/audit/upper-military-reserve-fat-tire-traverse-fd06a.gpx",
  "gpx/official/dry-creek-15a-1.gpx",
  "gpx/cues/dry-creek-15a-1.gpx",
  "gpx/audit/dry-creek-15a-1.gpx",
  "gpx/official/rocky-canyon-orchard-gulch-14.gpx",
  "gpx/cues/rocky-canyon-orchard-gulch-14.gpx",
  "gpx/audit/rocky-canyon-orchard-gulch-14.gpx",
  "gpx/official/dry-creek-sweet-connie-16a-1.gpx",
  "gpx/cues/dry-creek-sweet-connie-16a-1.gpx",
  "gpx/audit/dry-creek-sweet-connie-16a-1.gpx",
  "gpx/official/military-reserve-military-reserve-connection-fd15a.gpx",
  "gpx/cues/military-reserve-military-reserve-connection-fd15a.gpx",
  "gpx/audit/military-reserve-military-reserve-connection-fd15a.gpx",
  "gpx/official/polecat-gulch-polecat-loop-fd18a.gpx",
  "gpx/cues/polecat-gulch-polecat-loop-fd18a.gpx",
  "gpx/audit/polecat-gulch-polecat-loop-fd18a.gpx",
  "gpx/official/rocky-canyon-three-bears-fd20a.gpx",
  "gpx/cues/rocky-canyon-three-bears-fd20a.gpx",
  "gpx/audit/rocky-canyon-three-bears-fd20a.gpx",
  "gpx/official/upper-hulls-gulch-8th-street-motorcycle-fd23a.gpx",
  "gpx/cues/upper-hulls-gulch-8th-street-motorcycle-fd23a.gpx",
  "gpx/audit/upper-hulls-gulch-8th-street-motorcycle-fd23a.gpx",
  "gpx/official/bogus-basin-around-the-mountain-fd26a.gpx",
  "gpx/cues/bogus-basin-around-the-mountain-fd26a.gpx",
  "gpx/audit/bogus-basin-around-the-mountain-fd26a.gpx",
  "gpx/official/avimor-harlow-twisted-spring-h1.gpx",
  "gpx/cues/avimor-harlow-twisted-spring-h1.gpx",
  "gpx/audit/avimor-harlow-twisted-spring-h1.gpx",
  "gpx/official/bogus-basin-brewers-byway-ext-18.gpx",
  "gpx/cues/bogus-basin-brewers-byway-ext-18.gpx",
  "gpx/audit/bogus-basin-brewers-byway-ext-18.gpx"
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
