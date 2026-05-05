const CACHE_NAME = "boise-trails-field-packet-v23-496fa5819416";
const PRECACHE_URLS = [
  "./",
  "index.html",
  "manifest.json",
  "manifest.webmanifest",
  "icons/icon-192.png",
  "icons/icon-512.png",
  "gpx/all-field-packet-gpx.zip",
  "gpx/4b-upper-interpretive-scott-s-trail.gpx",
  "gpx/4a-bob-s-bob-s-trail-urban-connector.gpx",
  "gpx/1b-harrison-hollow-who-now-loop-trail-harrison-ridge-harrison-hollow-kemper-s-ridge-trail-.gpx",
  "gpx/7-seamans-gulch-seaman-gulch-trail-wild-phlox-trail.gpx",
  "gpx/8-homestead-harris-ridge-trail-peace-valley-overlook.gpx",
  "gpx/16b-freddy-s-stack-rock-stack-rock-connector.gpx",
  "gpx/11-hawkins-range-reserve-hawkins.gpx",
  "gpx/15b-dry-creek-parking-area-trailhead-red-tail-trail-landslide.gpx",
  "gpx/1a-west-climb-full-sail-trail-bob-smylie-buena-vista-trail-36th-street-chute.gpx",
  "gpx/9-veterans-veterans-big-springs-rabbit-run-d-s-chaos-rei-connection.gpx",
  "gpx/19-cervidae-arrow-rock-road-osm-parking-cervidae-peak.gpx",
  "gpx/14-orchard-gulch-orchard-gulch-trail-five-mile-gulch-trail-watchman-trail.gpx",
  "gpx/3-freestone-creek-military-reserve-connection-mountain-cove-central-ridge-trail-central-ri.gpx",
  "gpx/4c-warm-springs-golf-course-tram-trail-rock-island-rock-garden-table-rock-trail-quarry-tra.gpx",
  "gpx/12-8th-street-atv-parking-area-8th-street-motorcycle-trail-sidewinder-trail-corrals-trail.gpx",
  "gpx/5-cartwright-polecat-loop-doe-ridge-quick-draw-barn-owl.gpx",
  "gpx/2-hulls-gulch-lower-hull-s-gulch-trail-hull-s-gulch-interpretive-crestline-trail-red-cliff.gpx",
  "gpx/18-pioneer-lodge-parking-area-brewer-s-byway-extension-brewers-byway-shindig-tempest-trail.gpx",
  "gpx/15a-millergulch-parking-area-trailhead-connector-highlands-trail-dry-creek-trail.gpx",
  "gpx/17-simplot-lodge-parking-area-sunshine-xc-deer-point-trail-around-the-mountain-trail-the-f.gpx",
  "gpx/6-cartwright-peggy-s-trail-chukar-butte-trail-cartwright-connector-cartwright-ridge-chbh-c.gpx",
  "gpx/13-freestone-creek-three-bears-trail-femrite-s-patrol-freestone-ridge-two-point-shane-s-tr.gpx",
  "gpx/10-dry-creek-parking-area-trailhead-bitterbrush-trail-currant-creek-harlow-s-hollows-harlo.gpx"
];

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
