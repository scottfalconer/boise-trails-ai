const CACHE_NAME = "boise-trails-field-packet-v44-e65eb286c8e36545ca";
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
  "gpx/official/fd28a-millergulch-parking-area-trailhead-connector.gpx",
  "gpx/cues/fd28a-millergulch-parking-area-trailhead-connector.gpx",
  "gpx/audit/fd28a-millergulch-parking-area-trailhead-connector.gpx",
  "gpx/official/fd19a-hulls-gulch-kestral-trail.gpx",
  "gpx/cues/fd19a-hulls-gulch-kestral-trail.gpx",
  "gpx/audit/fd19a-hulls-gulch-kestral-trail.gpx",
  "gpx/official/fd14a-cartwright-doe-ridge.gpx",
  "gpx/cues/fd14a-cartwright-doe-ridge.gpx",
  "gpx/audit/fd14a-cartwright-doe-ridge.gpx",
  "gpx/official/fd14d-full-sail-36th-street-chute.gpx",
  "gpx/cues/fd14d-full-sail-36th-street-chute.gpx",
  "gpx/audit/fd14d-full-sail-36th-street-chute.gpx",
  "gpx/official/4b-upper-interpretive-scott-s-trail.gpx",
  "gpx/cues/4b-upper-interpretive-scott-s-trail.gpx",
  "gpx/audit/4b-upper-interpretive-scott-s-trail.gpx",
  "gpx/official/fd22c-the-grove-owl-s-roost-chickadee-ridge-trail-15th-st-trail-gold-finch.gpx",
  "gpx/cues/fd22c-the-grove-owl-s-roost-chickadee-ridge-trail-15th-st-trail-gold-finch.gpx",
  "gpx/audit/fd22c-the-grove-owl-s-roost-chickadee-ridge-trail-15th-st-trail-gold-finch.gpx",
  "gpx/official/4a-bob-s-bob-s-trail-urban-connector.gpx",
  "gpx/cues/4a-bob-s-bob-s-trail-urban-connector.gpx",
  "gpx/audit/4a-bob-s-bob-s-trail-urban-connector.gpx",
  "gpx/official/fd14b-cartwright-chbh-connector-quick-draw.gpx",
  "gpx/cues/fd14b-cartwright-chbh-connector-quick-draw.gpx",
  "gpx/audit/fd14b-cartwright-chbh-connector-quick-draw.gpx",
  "gpx/official/fd19b-hulls-gulch-lower-hull-s-gulch-trail-red-cliffs.gpx",
  "gpx/cues/fd19b-hulls-gulch-lower-hull-s-gulch-trail-red-cliffs.gpx",
  "gpx/audit/fd19b-hulls-gulch-lower-hull-s-gulch-trail-red-cliffs.gpx",
  "gpx/official/fd22b-hulls-gulch-crestline-trail.gpx",
  "gpx/cues/fd22b-hulls-gulch-crestline-trail.gpx",
  "gpx/audit/fd22b-hulls-gulch-crestline-trail.gpx",
  "gpx/official/16a-2-dry-creek-sweet-connie-roadside-parking-sheep-camp-trail.gpx",
  "gpx/cues/16a-2-dry-creek-sweet-connie-roadside-parking-sheep-camp-trail.gpx",
  "gpx/audit/16a-2-dry-creek-sweet-connie-roadside-parking-sheep-camp-trail.gpx",
  "gpx/official/fd19c-freestone-creek-shane-s-trail.gpx",
  "gpx/cues/fd19c-freestone-creek-shane-s-trail.gpx",
  "gpx/audit/fd19c-freestone-creek-shane-s-trail.gpx",
  "gpx/official/fd21b-old-pen-table-rock-trail-quarry-trail-castle-rock-shoshone-paiute.gpx",
  "gpx/cues/fd21b-old-pen-table-rock-trail-quarry-trail-castle-rock-shoshone-paiute.gpx",
  "gpx/audit/fd21b-old-pen-table-rock-trail-quarry-trail-castle-rock-shoshone-paiute.gpx",
  "gpx/official/fd07a-simplot-lodge-parking-area-sunshine-xc.gpx",
  "gpx/cues/fd07a-simplot-lodge-parking-area-sunshine-xc.gpx",
  "gpx/audit/fd07a-simplot-lodge-parking-area-sunshine-xc.gpx",
  "gpx/official/fd08a-cartwright-cartwright-ridge.gpx",
  "gpx/cues/fd08a-cartwright-cartwright-ridge.gpx",
  "gpx/audit/fd08a-cartwright-cartwright-ridge.gpx",
  "gpx/official/7-seamans-gulch-seaman-gulch-trail-wild-phlox-trail.gpx",
  "gpx/cues/7-seamans-gulch-seaman-gulch-trail-wild-phlox-trail.gpx",
  "gpx/audit/7-seamans-gulch-seaman-gulch-trail-wild-phlox-trail.gpx",
  "gpx/official/fd08b-cartwright-cartwright-connector.gpx",
  "gpx/cues/fd08b-cartwright-cartwright-connector.gpx",
  "gpx/audit/fd08b-cartwright-cartwright-connector.gpx",
  "gpx/official/16b-freddy-s-stack-rock-stack-rock-connector.gpx",
  "gpx/cues/16b-freddy-s-stack-rock-stack-rock-connector.gpx",
  "gpx/audit/16b-freddy-s-stack-rock-stack-rock-connector.gpx",
  "gpx/official/fd05a-8th-street-atv-parking-area-hull-s-gulch-interpretive.gpx",
  "gpx/cues/fd05a-8th-street-atv-parking-area-hull-s-gulch-interpretive.gpx",
  "gpx/audit/fd05a-8th-street-atv-parking-area-hull-s-gulch-interpretive.gpx",
  "gpx/official/fd09a-dry-creek-parking-area-trailhead-barn-owl.gpx",
  "gpx/cues/fd09a-dry-creek-parking-area-trailhead-barn-owl.gpx",
  "gpx/audit/fd09a-dry-creek-parking-area-trailhead-barn-owl.gpx",
  "gpx/official/15b-dry-creek-parking-area-trailhead-red-tail-trail-landslide.gpx",
  "gpx/cues/15b-dry-creek-parking-area-trailhead-red-tail-trail-landslide.gpx",
  "gpx/audit/15b-dry-creek-parking-area-trailhead-red-tail-trail-landslide.gpx",
  "gpx/official/11-hawkins-range-reserve-hawkins.gpx",
  "gpx/cues/11-hawkins-range-reserve-hawkins.gpx",
  "gpx/audit/11-hawkins-range-reserve-hawkins.gpx",
  "gpx/official/10b-dry-creek-parking-area-trailhead-bitterbrush-trail-currant-creek.gpx",
  "gpx/cues/10b-dry-creek-parking-area-trailhead-bitterbrush-trail-currant-creek.gpx",
  "gpx/audit/10b-dry-creek-parking-area-trailhead-bitterbrush-trail-currant-creek.gpx",
  "gpx/official/fd07b-simplot-lodge-parking-area-deer-point-trail.gpx",
  "gpx/cues/fd07b-simplot-lodge-parking-area-deer-point-trail.gpx",
  "gpx/audit/fd07b-simplot-lodge-parking-area-deer-point-trail.gpx",
  "gpx/official/fd21a-homestead-peace-valley-overlook-harris-ridge-trail.gpx",
  "gpx/cues/fd21a-homestead-peace-valley-overlook-harris-ridge-trail.gpx",
  "gpx/audit/fd21a-homestead-peace-valley-overlook-harris-ridge-trail.gpx",
  "gpx/official/fd25b-pioneer-lodge-parking-area-the-face-trail.gpx",
  "gpx/cues/fd25b-pioneer-lodge-parking-area-the-face-trail.gpx",
  "gpx/audit/fd25b-pioneer-lodge-parking-area-the-face-trail.gpx",
  "gpx/official/fd01a-warm-springs-golf-course-rock-island-rock-garden-tram-trail-table-rock-quarry-trail.gpx",
  "gpx/cues/fd01a-warm-springs-golf-course-rock-island-rock-garden-tram-trail-table-rock-quarry-trail.gpx",
  "gpx/audit/fd01a-warm-springs-golf-course-rock-island-rock-garden-tram-trail-table-rock-quarry-trail.gpx",
  "gpx/official/fd25a-simplot-lodge-parking-area-elk-meadows-trail.gpx",
  "gpx/cues/fd25a-simplot-lodge-parking-area-elk-meadows-trail.gpx",
  "gpx/audit/fd25a-simplot-lodge-parking-area-elk-meadows-trail.gpx",
  "gpx/official/9-veterans-veterans-big-springs-rabbit-run-d-s-chaos-rei-connection.gpx",
  "gpx/cues/9-veterans-veterans-big-springs-rabbit-run-d-s-chaos-rei-connection.gpx",
  "gpx/audit/9-veterans-veterans-big-springs-rabbit-run-d-s-chaos-rei-connection.gpx",
  "gpx/official/fd03a-dry-creek-parking-area-trailhead-chukar-butte-trail.gpx",
  "gpx/cues/fd03a-dry-creek-parking-area-trailhead-chukar-butte-trail.gpx",
  "gpx/audit/fd03a-dry-creek-parking-area-trailhead-chukar-butte-trail.gpx",
  "gpx/official/19-cervidae-arrow-rock-road-osm-parking-cervidae-peak.gpx",
  "gpx/cues/19-cervidae-arrow-rock-road-osm-parking-cervidae-peak.gpx",
  "gpx/audit/19-cervidae-arrow-rock-road-osm-parking-cervidae-peak.gpx",
  "gpx/official/fd04a-freestone-creek-two-point-shane-s-connector-femrite-s-patrol.gpx",
  "gpx/cues/fd04a-freestone-creek-two-point-shane-s-connector-femrite-s-patrol.gpx",
  "gpx/audit/fd04a-freestone-creek-two-point-shane-s-connector-femrite-s-patrol.gpx",
  "gpx/official/fd06a-lower-interpretive-fat-tire-traverse-curlew-connection.gpx",
  "gpx/cues/fd06a-lower-interpretive-fat-tire-traverse-curlew-connection.gpx",
  "gpx/audit/fd06a-lower-interpretive-fat-tire-traverse-curlew-connection.gpx",
  "gpx/official/15a-1-dry-creek-sweet-connie-roadside-parking-dry-creek-trail-shingle-creek-trail.gpx",
  "gpx/cues/15a-1-dry-creek-sweet-connie-roadside-parking-dry-creek-trail-shingle-creek-trail.gpx",
  "gpx/audit/15a-1-dry-creek-sweet-connie-roadside-parking-dry-creek-trail-shingle-creek-trail.gpx",
  "gpx/official/14-orchard-gulch-orchard-gulch-trail-five-mile-gulch-trail-watchman-trail.gpx",
  "gpx/cues/14-orchard-gulch-orchard-gulch-trail-five-mile-gulch-trail-watchman-trail.gpx",
  "gpx/audit/14-orchard-gulch-orchard-gulch-trail-five-mile-gulch-trail-watchman-trail.gpx",
  "gpx/official/fd12a-west-climb-who-now-loop-trail-harrison-ridge-harrison-hollow-kemper-s-ridge-trail-fu.gpx",
  "gpx/cues/fd12a-west-climb-who-now-loop-trail-harrison-ridge-harrison-hollow-kemper-s-ridge-trail-fu.gpx",
  "gpx/audit/fd12a-west-climb-who-now-loop-trail-harrison-ridge-harrison-hollow-kemper-s-ridge-trail-fu.gpx",
  "gpx/official/16a-1-dry-creek-sweet-connie-roadside-parking-sweet-connie-trail.gpx",
  "gpx/cues/16a-1-dry-creek-sweet-connie-roadside-parking-sweet-connie-trail.gpx",
  "gpx/audit/16a-1-dry-creek-sweet-connie-roadside-parking-sweet-connie-trail.gpx",
  "gpx/official/3-freestone-creek-military-reserve-connection-mountain-cove-central-ridge-trail-central-ri.gpx",
  "gpx/cues/3-freestone-creek-military-reserve-connection-mountain-cove-central-ridge-trail-central-ri.gpx",
  "gpx/audit/3-freestone-creek-military-reserve-connection-mountain-cove-central-ridge-trail-central-ri.gpx",
  "gpx/official/fd18a-cartwright-polecat-loop-peggy-s-trail.gpx",
  "gpx/cues/fd18a-cartwright-polecat-loop-peggy-s-trail.gpx",
  "gpx/audit/fd18a-cartwright-polecat-loop-peggy-s-trail.gpx",
  "gpx/official/fd20a-freestone-creek-three-bears-trail-freestone-ridge.gpx",
  "gpx/cues/fd20a-freestone-creek-three-bears-trail-freestone-ridge.gpx",
  "gpx/audit/fd20a-freestone-creek-three-bears-trail-freestone-ridge.gpx",
  "gpx/official/12-8th-street-atv-parking-area-8th-street-motorcycle-trail-sidewinder-trail-corrals-trail-.gpx",
  "gpx/cues/12-8th-street-atv-parking-area-8th-street-motorcycle-trail-sidewinder-trail-corrals-trail-.gpx",
  "gpx/audit/12-8th-street-atv-parking-area-8th-street-motorcycle-trail-sidewinder-trail-corrals-trail-.gpx",
  "gpx/official/fd26a-simplot-lodge-parking-area-around-the-mountain-trail.gpx",
  "gpx/cues/fd26a-simplot-lodge-parking-area-around-the-mountain-trail.gpx",
  "gpx/audit/fd26a-simplot-lodge-parking-area-around-the-mountain-trail.gpx",
  "gpx/official/h1-avimor-spring-valley-creek-parking-twisted-spring-ricochet-shooting-range-whistling-pig.gpx",
  "gpx/cues/h1-avimor-spring-valley-creek-parking-twisted-spring-ricochet-shooting-range-whistling-pig.gpx",
  "gpx/audit/h1-avimor-spring-valley-creek-parking-twisted-spring-ricochet-shooting-range-whistling-pig.gpx",
  "gpx/official/18-pioneer-lodge-parking-area-brewer-s-byway-extension-brewers-byway-shindig-tempest-trail.gpx",
  "gpx/cues/18-pioneer-lodge-parking-area-brewer-s-byway-extension-brewers-byway-shindig-tempest-trail.gpx",
  "gpx/audit/18-pioneer-lodge-parking-area-brewer-s-byway-extension-brewers-byway-shindig-tempest-trail.gpx"
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
