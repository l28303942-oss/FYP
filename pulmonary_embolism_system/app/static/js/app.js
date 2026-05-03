/**
 * Global UI: theme, navbar, modals, toasts, dashboard counters, patient edit.
 */
(function () {
  const root = document.documentElement;

  function getStoredTheme() {
    return localStorage.getItem("pe-theme");
  }

  function applyTheme(mode) {
    if (!mode) return;
    root.dataset.theme = mode;
    localStorage.setItem("pe-theme", mode);
  }

  const stored = getStoredTheme();
  if (stored && (stored === "dark" || stored === "light")) {
    root.dataset.theme = stored;
  }

  document.querySelectorAll("#themeToggleDash, [data-theme-toggle]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const next = root.dataset.theme === "dark" ? "light" : "dark";
      applyTheme(next);
    });
  });

  const sidebar = document.getElementById("sidebar");
  const sidebarToggle = document.getElementById("sidebarToggle");
  const sidebarBackdrop = document.getElementById("sidebarBackdrop");

  function closeSidebar() {
    sidebar?.classList.remove("open");
    sidebarBackdrop?.classList.remove("visible");
    sidebarBackdrop?.setAttribute("aria-hidden", "true");
  }

  function openSidebar() {
    sidebar?.classList.add("open");
    sidebarBackdrop?.classList.add("visible");
    sidebarBackdrop?.setAttribute("aria-hidden", "false");
  }

  if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener("click", () => {
      if (sidebar.classList.contains("open")) closeSidebar();
      else openSidebar();
    });
  }
  if (sidebarBackdrop) {
    sidebarBackdrop.addEventListener("click", closeSidebar);
  }
  document.querySelectorAll(".sidebar-nav a").forEach((a) => {
    a.addEventListener("click", () => {
      if (window.matchMedia("(max-width: 960px)").matches) closeSidebar();
    });
  });

  document.querySelectorAll("[data-open]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.getAttribute("data-open");
      const dlg = document.getElementById(id);
      if (dlg && typeof dlg.showModal === "function") dlg.showModal();
    });
  });

  document.querySelectorAll("[data-close]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.getAttribute("data-close");
      const dlg = document.getElementById(id);
      if (dlg && typeof dlg.close === "function") dlg.close();
    });
  });

  document.querySelectorAll('[data-edit-patient]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const id = btn.dataset.id;
      const form = document.getElementById('editPatientForm');
      if (!form) return;
      form.action = '/patients/' + id + '/edit';
      const setVal = (sel, v) => {
        const el = document.querySelector(sel);
        if (el) el.value = v ?? '';
      };
      setVal('#ep_name', btn.dataset.name);
      setVal('#ep_age', btn.dataset.age);
      setVal('#ep_contact', btn.dataset.contact);
      setVal('#ep_notes', btn.dataset.notes);
      setVal('#ep_scan', btn.dataset.scan);
      setVal('#ep_doctor', btn.dataset.doctor);
      const g = document.getElementById('ep_gender');
      if (g) g.value = btn.dataset.gender || 'Other';
      const dlg = document.getElementById('editPatient');
      if (dlg && typeof dlg.showModal === 'function') dlg.showModal();
    });
  });

  function animateCount(el) {
    const target = parseFloat(el.dataset.count);
    if (isNaN(target)) return;
    const duration = 900;
    const t0 = performance.now();
    function step(now) {
      const p = Math.min(1, (now - t0) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      const val = Math.round(target * eased * 100) / 100;
      el.textContent = Number.isInteger(target) ? Math.round(val) : val.toFixed(1);
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  document.querySelectorAll('[data-count]').forEach(animateCount);

  document.querySelectorAll('.toast').forEach((t) => {
    setTimeout(() => {
      t.style.opacity = '0';
      t.style.transform = 'translateX(8px)';
      setTimeout(() => t.remove(), 400);
    }, 4200);
  });

  document.querySelectorAll('.print-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const url = btn.getAttribute('data-href');
      if (url) window.open(url, '_blank');
    });
  });
})();
