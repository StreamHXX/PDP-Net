"use strict";
(() => {
  const config = window.PDP_PAGE || {};
  for (const key of ["authors", "affiliations"]) {
    const element = document.getElementById(key);
    if (typeof config[key] === "string" && config[key].trim()) {
      element.textContent = config[key].trim();
      element.hidden = false;
    }
  }
  document.querySelectorAll("[data-resource]").forEach(link => {
    const value = config[link.dataset.resource];
    if (typeof value !== "string") return;
    try {
      const url = new URL(value);
      if (url.protocol === "https:" || url.protocol === "http:") link.href = url.href;
    } catch { /* Retain the static link if a configuration value is malformed. */ }
  });
})();
