// Client-side interactions for the Churn Prediction System
document.addEventListener("DOMContentLoaded", function () {

    /* ---- Button loading state on any form submit ---- */
    document.querySelectorAll("form").forEach(function (form) {
        form.addEventListener("submit", function () {
            const btn = form.querySelector("button[type='submit']");
            if (btn && !btn.classList.contains("is-loading")) {
                btn.classList.add("is-loading");
                btn.setAttribute("disabled", "disabled");
                const label = btn.querySelector(".btn-label");
                if (label) label.setAttribute("data-loading", "Processing\u2026");
            }
        });
    });

    /* ---- KPI count-up animation (dashboard) ---- */
    const counters = document.querySelectorAll(".count-up");
    if (counters.length) {
        counters.forEach(function (el) {
            const target = parseFloat(el.getAttribute("data-count")) || 0;
            const decimals = parseInt(el.getAttribute("data-decimals") || "0", 10);
            const suffix = el.getAttribute("data-suffix") || "";
            const duration = 900;
            const start = performance.now();

            function tick(now) {
                const progress = Math.min((now - start) / duration, 1);
                const eased = 1 - Math.pow(1 - progress, 3);
                const value = target * eased;
                el.textContent = value.toFixed(decimals) + suffix;
                if (progress < 1) requestAnimationFrame(tick);
                else el.textContent = target.toFixed(decimals) + suffix;
            }
            requestAnimationFrame(tick);
        });
    }

    /* ---- Batch upload: dropzone, file chip, fake progress on submit ---- */
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("dataset-input");
    const fileChip = document.getElementById("file-chip");
    const fileNameEl = document.getElementById("file-name");
    const fileSizeEl = document.getElementById("file-size");
    const batchForm = document.getElementById("batch-form");
    const progressTrack = document.getElementById("progress-track");
    const progressBar = document.getElementById("progress-bar");

    function showFile(file) {
        if (!file) return;
        fileNameEl.textContent = file.name;
        fileSizeEl.textContent = (file.size / 1024).toFixed(1) + " KB";
        fileChip.classList.add("show");
    }

    if (dropzone && fileInput) {
        fileInput.addEventListener("change", function () {
            if (fileInput.files && fileInput.files[0]) showFile(fileInput.files[0]);
        });

        ["dragenter", "dragover"].forEach(function (evt) {
            dropzone.addEventListener(evt, function (e) {
                e.preventDefault();
                dropzone.classList.add("drag-over");
            });
        });
        ["dragleave", "drop"].forEach(function (evt) {
            dropzone.addEventListener(evt, function (e) {
                e.preventDefault();
                dropzone.classList.remove("drag-over");
            });
        });
        dropzone.addEventListener("drop", function (e) {
            const dt = e.dataTransfer;
            if (dt && dt.files && dt.files[0]) {
                fileInput.files = dt.files;
                showFile(dt.files[0]);
            }
        });
    }

    if (batchForm && progressTrack && progressBar) {
        batchForm.addEventListener("submit", function () {
            progressTrack.classList.add("show");
            progressBar.style.width = "12%";
            setTimeout(function () { progressBar.style.width = "55%"; }, 150);
            setTimeout(function () { progressBar.style.width = "85%"; }, 500);
        });
    }
});
