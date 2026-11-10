/* Meliona's service worker: one job only. A page opened with no connection
 * gets the offline page instead of the browser's own error screen
 * (John, 2026-11-10: offline support). Every other request goes straight to
 * the network untouched - nothing is served stale, so a deploy is never
 * hidden behind an old copy. Registered by static/dock.js. */
var CACHE = 'meliona-offline-v1';
var OFFLINE = '/offline.html';

self.addEventListener('install', function (e) {
  e.waitUntil(caches.open(CACHE).then(function (c) { return c.addAll([OFFLINE, '/favicon.png']); })
    .then(function () { return self.skipWaiting(); }));
});

self.addEventListener('activate', function (e) {
  e.waitUntil(caches.keys().then(function (keys) {
    return Promise.all(keys.filter(function (k) { return k !== CACHE; })
      .map(function (k) { return caches.delete(k); }));
  }).then(function () { return self.clients.claim(); }));
});

self.addEventListener('fetch', function (e) {
  if (e.request.mode !== 'navigate') return;           // only a page load, never data or the socket
  e.respondWith(fetch(e.request).catch(function () {
    return caches.match(OFFLINE);
  }));
});
