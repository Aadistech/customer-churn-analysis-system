// Client-side interactions for Enterprise Churn System
document.addEventListener("DOMContentLoaded", function() {
    const form = document.querySelector("form");
    if (form) {
        form.addEventListener("submit", function() {
            const btn = form.querySelector("button[type='submit']");
            if (btn) {
                btn.textContent = "Processing Telemetry...";
                btn.style.opacity = "0.7";
            }
        });
    }
});