/**
 * Minimal interactive atlas (Leaflet + Esri tiled MapServer).
 * Config: docs/config/map-config.json (same idea as webmap-demo app-config.json).
 */
(function () {
  const host = document.getElementById("atlas-map-host");
  if (!host) return;

  const els = {
    map: host.querySelector("#map"),
    status: host.querySelector(".map-status"),
    naip: host.querySelector("#layer-naip"),
    prediction: host.querySelector("#layer-prediction"),
    gt: host.querySelector("#layer-gt"),
    boundary: host.querySelector("#layer-boundary"),
    opacity: host.querySelector("#prediction-opacity"),
    opacityValue: host.querySelector("#prediction-opacity-value"),
    gtRow: host.querySelector("[data-layer-row='gt']"),
    boundaryRow: host.querySelector("[data-layer-row='boundary']"),
  };

  let map = null;
  const layerRefs = { naip: null, prediction: null, gt: null, boundary: null };

  function hasUrl(url) {
    return typeof url === "string" && url.trim().length > 0;
  }

  function setStatus(message, isError) {
    if (!els.status) return;
    els.status.textContent = message || "";
    els.status.hidden = !message;
    els.status.classList.toggle("map-status--error", Boolean(isError));
  }

  function addTiledLayer(url, options) {
    if (!hasUrl(url) || typeof L === "undefined" || !L.esri) return null;
    const layer = L.esri.tiledMapLayer({
      url: url.trim(),
      opacity: options.opacity ?? 1,
      maxZoom: 19,
    });
    if (options.visible !== false) {
      layer.addTo(map);
    }
    return layer;
  }

  function bindLayerToggle(checkbox, layerKey) {
    if (!checkbox) return;
    checkbox.addEventListener("change", () => {
      const layer = layerRefs[layerKey];
      if (!layer || !map) return;
      if (checkbox.checked) {
        layer.addTo(map);
      } else {
        map.removeLayer(layer);
      }
    });
  }

  function initControls(config) {
    const predOpacity = config.prediction?.opacity ?? 0.55;

    if (els.naip) els.naip.checked = true;
    if (els.prediction) els.prediction.checked = true;
    if (els.opacity) els.opacity.value = String(predOpacity);
    if (els.opacityValue) els.opacityValue.textContent = predOpacity.toFixed(2);

    if (els.gtRow) {
      const show = hasUrl(config.layers?.groundTruthUrl);
      els.gtRow.hidden = !show;
      if (els.gt) {
        els.gt.disabled = !show;
        if (show) els.gt.checked = false;
      }
    }
    if (els.boundaryRow) {
      const show = hasUrl(config.layers?.tileBoundaryUrl);
      els.boundaryRow.hidden = !show;
      if (els.boundary) {
        els.boundary.disabled = !show;
        if (show) els.boundary.checked = false;
      }
    }

    bindLayerToggle(els.naip, "naip");
    bindLayerToggle(els.prediction, "prediction");
    bindLayerToggle(els.gt, "gt");
    bindLayerToggle(els.boundary, "boundary");

    if (els.opacity) {
      const updateOpacity = () => {
        const value = Number(els.opacity.value);
        if (els.opacityValue) els.opacityValue.textContent = value.toFixed(2);
        if (layerRefs.prediction) layerRefs.prediction.setOpacity(value);
      };
      els.opacity.addEventListener("input", updateOpacity);
      updateOpacity();
    }
  }

  async function start() {
    setStatus("Loading map…", false);

    let config;
    try {
      const res = await fetch("config/map-config.json");
      if (!res.ok) throw new Error("config not found");
      config = await res.json();
    } catch (e) {
      setStatus("Failed to load map config.", true);
      return;
    }

    if (typeof L === "undefined" || !L.esri) {
      setStatus("Map libraries failed to load.", true);
      return;
    }

    const center = config.map?.center ?? [39.15, -77.05];
    const zoom = config.map?.zoom ?? 10;
    const predOpacity = config.prediction?.opacity ?? 0.55;

    map = L.map(els.map, {
      center,
      zoom,
      zoomControl: true,
      attributionControl: true,
    });

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>',
    }).addTo(map);

    const naipUrl = config.layers?.naipUrl;
    const predictionUrl = config.layers?.predictionUrl;

    if (hasUrl(naipUrl)) {
      layerRefs.naip = addTiledLayer(naipUrl, { opacity: 1, visible: true });
    }
    if (hasUrl(predictionUrl)) {
      layerRefs.prediction = addTiledLayer(predictionUrl, {
        opacity: predOpacity,
        visible: true,
      });
    }

    if (hasUrl(config.layers?.groundTruthUrl)) {
      layerRefs.gt = addTiledLayer(config.layers.groundTruthUrl, { visible: false });
    }
    if (hasUrl(config.layers?.tileBoundaryUrl)) {
      layerRefs.boundary = addTiledLayer(config.layers.tileBoundaryUrl, { visible: false });
    }

    initControls(config);

    if (!layerRefs.naip && !layerRefs.prediction) {
      setStatus("No imagery URLs configured. Edit docs/config/map-config.json.", true);
      return;
    }

    setStatus("", false);
    host.classList.add("map-host--ready");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
