(function () {
  const dz = document.getElementById('dropzone');
  const input = document.getElementById('fileInput');
  const form = document.getElementById('uploadForm');
  const progressWrap = document.getElementById('progressWrap');
  const progressFill = document.getElementById('progressFill');
  const progressLabel = document.getElementById('progressLabel');
  const patientSelect = document.getElementById('patientSelect');

  if (!dz || !input || !form) return;

  dz.addEventListener('click', () => input.click());

  ['dragenter', 'dragover'].forEach((ev) => {
    dz.addEventListener(ev, (e) => {
      e.preventDefault();
      dz.classList.add('dragover');
    });
  });
  ['dragleave', 'drop'].forEach((ev) => {
    dz.addEventListener(ev, (e) => {
      e.preventDefault();
      dz.classList.remove('dragover');
    });
  });
  dz.addEventListener('drop', (e) => {
    const f = e.dataTransfer.files[0];
    if (f) {
      input.files = e.dataTransfer.files;
      dz.querySelector('.dz-inner p strong').textContent = f.name;
    }
  });
  input.addEventListener('change', () => {
    const f = input.files[0];
    if (f) dz.querySelector('.dz-inner p strong').textContent = f.name;
  });

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    if (!patientSelect.value || !input.files.length) {
      alert('Select patient and file.');
      return;
    }
    const fd = new FormData();
    fd.append('patient_id', patientSelect.value);
    fd.append('scan_file', input.files[0]);

    progressWrap.hidden = false;
    progressFill.style.width = '0%';
    progressLabel.textContent = '0%';

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/scan/upload');
    xhr.upload.onprogress = (ev) => {
      if (ev.lengthComputable) {
        const pct = Math.round((ev.loaded / ev.total) * 100);
        progressFill.style.width = pct + '%';
        progressLabel.textContent = pct + '%';
      }
    };
    xhr.onload = () => {
      try {
        const res = JSON.parse(xhr.responseText);
        if (res.ok) {
          window.location.href = '/segmentation';
        } else {
          alert(res.error || 'Upload failed');
        }
      } catch {
        alert('Upload failed');
      }
    };
    xhr.onerror = () => alert('Network error');
    xhr.send(fd);
  });
})();
