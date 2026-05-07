const CACHE_NAME = "boise-trails-field-packet-v27-633ff6b16f1dc46974";
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
  "gpx/navigation/4b-upper-interpretive-scott-s-trail.gpx",
  "gpx/cues/4b-upper-interpretive-scott-s-trail.gpx",
  "gpx/audit/4b-upper-interpretive-scott-s-trail.gpx",
  "gpx/navigation/4a-bob-s-bob-s-trail-urban-connector.gpx",
  "gpx/cues/4a-bob-s-bob-s-trail-urban-connector.gpx",
  "gpx/audit/4a-bob-s-bob-s-trail-urban-connector.gpx",
  "gpx/navigation/8b-homestead-peace-valley-overlook.gpx",
  "gpx/cues/8b-homestead-peace-valley-overlook.gpx",
  "gpx/audit/8b-homestead-peace-valley-overlook.gpx",
  "gpx/navigation/8a-homestead-harris-ridge-trail.gpx",
  "gpx/cues/8a-homestead-harris-ridge-trail.gpx",
  "gpx/audit/8a-homestead-harris-ridge-trail.gpx",
  "gpx/navigation/7-seamans-gulch-seaman-gulch-trail-wild-phlox-trail.gpx",
  "gpx/cues/7-seamans-gulch-seaman-gulch-trail-wild-phlox-trail.gpx",
  "gpx/audit/7-seamans-gulch-seaman-gulch-trail-wild-phlox-trail.gpx",
  "gpx/navigation/1a-west-climb-full-sail-trail-bob-smylie-buena-vista-trail-36th-street-chute.gpx",
  "gpx/cues/1a-west-climb-full-sail-trail-bob-smylie-buena-vista-trail-36th-street-chute.gpx",
  "gpx/audit/1a-west-climb-full-sail-trail-bob-smylie-buena-vista-trail-36th-street-chute.gpx",
  "gpx/navigation/16b-freddy-s-stack-rock-stack-rock-connector.gpx",
  "gpx/cues/16b-freddy-s-stack-rock-stack-rock-connector.gpx",
  "gpx/audit/16b-freddy-s-stack-rock-stack-rock-connector.gpx",
  "gpx/navigation/1b-harrison-hollow-who-now-loop-trail-harrison-ridge-harrison-hollow-kemper-s-ridge-trail-.gpx",
  "gpx/cues/1b-harrison-hollow-who-now-loop-trail-harrison-ridge-harrison-hollow-kemper-s-ridge-trail-.gpx",
  "gpx/audit/1b-harrison-hollow-who-now-loop-trail-harrison-ridge-harrison-hollow-kemper-s-ridge-trail-.gpx",
  "gpx/navigation/15b-dry-creek-parking-area-trailhead-red-tail-trail-landslide.gpx",
  "gpx/cues/15b-dry-creek-parking-area-trailhead-red-tail-trail-landslide.gpx",
  "gpx/audit/15b-dry-creek-parking-area-trailhead-red-tail-trail-landslide.gpx",
  "gpx/navigation/11-hawkins-range-reserve-hawkins.gpx",
  "gpx/cues/11-hawkins-range-reserve-hawkins.gpx",
  "gpx/audit/11-hawkins-range-reserve-hawkins.gpx",
  "gpx/navigation/10b-dry-creek-parking-area-trailhead-bitterbrush-trail-currant-creek.gpx",
  "gpx/cues/10b-dry-creek-parking-area-trailhead-bitterbrush-trail-currant-creek.gpx",
  "gpx/audit/10b-dry-creek-parking-area-trailhead-bitterbrush-trail-currant-creek.gpx",
  "gpx/navigation/9-veterans-veterans-big-springs-rabbit-run-d-s-chaos-rei-connection.gpx",
  "gpx/cues/9-veterans-veterans-big-springs-rabbit-run-d-s-chaos-rei-connection.gpx",
  "gpx/audit/9-veterans-veterans-big-springs-rabbit-run-d-s-chaos-rei-connection.gpx",
  "gpx/navigation/19-cervidae-arrow-rock-road-osm-parking-cervidae-peak.gpx",
  "gpx/cues/19-cervidae-arrow-rock-road-osm-parking-cervidae-peak.gpx",
  "gpx/audit/19-cervidae-arrow-rock-road-osm-parking-cervidae-peak.gpx",
  "gpx/navigation/14-orchard-gulch-orchard-gulch-trail-five-mile-gulch-trail-watchman-trail.gpx",
  "gpx/cues/14-orchard-gulch-orchard-gulch-trail-five-mile-gulch-trail-watchman-trail.gpx",
  "gpx/audit/14-orchard-gulch-orchard-gulch-trail-five-mile-gulch-trail-watchman-trail.gpx",
  "gpx/navigation/16a-1-dry-creek-sweet-connie-roadside-parking-sweet-connie-trail.gpx",
  "gpx/cues/16a-1-dry-creek-sweet-connie-roadside-parking-sweet-connie-trail.gpx",
  "gpx/audit/16a-1-dry-creek-sweet-connie-roadside-parking-sweet-connie-trail.gpx",
  "gpx/navigation/3-freestone-creek-military-reserve-connection-mountain-cove-central-ridge-trail-central-ri.gpx",
  "gpx/cues/3-freestone-creek-military-reserve-connection-mountain-cove-central-ridge-trail-central-ri.gpx",
  "gpx/audit/3-freestone-creek-military-reserve-connection-mountain-cove-central-ridge-trail-central-ri.gpx",
  "gpx/navigation/12-8th-street-atv-parking-area-8th-street-motorcycle-trail-sidewinder-trail-corrals-trail.gpx",
  "gpx/cues/12-8th-street-atv-parking-area-8th-street-motorcycle-trail-sidewinder-trail-corrals-trail.gpx",
  "gpx/audit/12-8th-street-atv-parking-area-8th-street-motorcycle-trail-sidewinder-trail-corrals-trail.gpx",
  "gpx/navigation/4c-eagle-rock-park-shoshone-paiute-quarry-trail-castle-rock-table-rock-quarry-trail-table-.gpx",
  "gpx/cues/4c-eagle-rock-park-shoshone-paiute-quarry-trail-castle-rock-table-rock-quarry-trail-table-.gpx",
  "gpx/audit/4c-eagle-rock-park-shoshone-paiute-quarry-trail-castle-rock-table-rock-quarry-trail-table-.gpx",
  "gpx/navigation/5-cartwright-polecat-loop-doe-ridge-quick-draw-barn-owl.gpx",
  "gpx/cues/5-cartwright-polecat-loop-doe-ridge-quick-draw-barn-owl.gpx",
  "gpx/audit/5-cartwright-polecat-loop-doe-ridge-quick-draw-barn-owl.gpx",
  "gpx/navigation/16a-2-dry-creek-sweet-connie-roadside-parking-sheep-camp-trail-shingle-creek-trail.gpx",
  "gpx/cues/16a-2-dry-creek-sweet-connie-roadside-parking-sheep-camp-trail-shingle-creek-trail.gpx",
  "gpx/audit/16a-2-dry-creek-sweet-connie-roadside-parking-sheep-camp-trail-shingle-creek-trail.gpx",
  "gpx/navigation/18-pioneer-lodge-parking-area-brewer-s-byway-extension-brewers-byway-shindig-tempest-trail.gpx",
  "gpx/cues/18-pioneer-lodge-parking-area-brewer-s-byway-extension-brewers-byway-shindig-tempest-trail.gpx",
  "gpx/audit/18-pioneer-lodge-parking-area-brewer-s-byway-extension-brewers-byway-shindig-tempest-trail.gpx",
  "gpx/navigation/2-hulls-gulch-lower-hull-s-gulch-trail-hull-s-gulch-interpretive-crestline-trail-red-cliff.gpx",
  "gpx/cues/2-hulls-gulch-lower-hull-s-gulch-trail-hull-s-gulch-interpretive-crestline-trail-red-cliff.gpx",
  "gpx/audit/2-hulls-gulch-lower-hull-s-gulch-trail-hull-s-gulch-interpretive-crestline-trail-red-cliff.gpx",
  "gpx/navigation/10a-harlow-s-hidden-springs-west-access-probe-harlow-s-hollows-harlow-s-hollows-connector-.gpx",
  "gpx/cues/10a-harlow-s-hidden-springs-west-access-probe-harlow-s-hollows-harlow-s-hollows-connector-.gpx",
  "gpx/audit/10a-harlow-s-hidden-springs-west-access-probe-harlow-s-hollows-harlow-s-hollows-connector-.gpx",
  "gpx/navigation/15a-millergulch-parking-area-trailhead-connector-highlands-trail-dry-creek-trail.gpx",
  "gpx/cues/15a-millergulch-parking-area-trailhead-connector-highlands-trail-dry-creek-trail.gpx",
  "gpx/audit/15a-millergulch-parking-area-trailhead-connector-highlands-trail-dry-creek-trail.gpx",
  "gpx/navigation/17-simplot-lodge-parking-area-sunshine-xc-deer-point-trail-around-the-mountain-trail-the-f.gpx",
  "gpx/cues/17-simplot-lodge-parking-area-sunshine-xc-deer-point-trail-around-the-mountain-trail-the-f.gpx",
  "gpx/audit/17-simplot-lodge-parking-area-sunshine-xc-deer-point-trail-around-the-mountain-trail-the-f.gpx",
  "gpx/navigation/6-cartwright-peggy-s-trail-chukar-butte-trail-cartwright-connector-cartwright-ridge-chbh-c.gpx",
  "gpx/cues/6-cartwright-peggy-s-trail-chukar-butte-trail-cartwright-connector-cartwright-ridge-chbh-c.gpx",
  "gpx/audit/6-cartwright-peggy-s-trail-chukar-butte-trail-cartwright-connector-cartwright-ridge-chbh-c.gpx",
  "gpx/navigation/13-freestone-creek-three-bears-trail-femrite-s-patrol-freestone-ridge-two-point-shane-s-tr.gpx",
  "gpx/cues/13-freestone-creek-three-bears-trail-femrite-s-patrol-freestone-ridge-two-point-shane-s-tr.gpx",
  "gpx/audit/13-freestone-creek-three-bears-trail-femrite-s-patrol-freestone-ridge-two-point-shane-s-tr.gpx"
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
