// Clicking anywhere on the install command copies it (through its copy button)
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('div.install-cmd').forEach((box) => {
        box.addEventListener('click', (event) => {
            const button = box.querySelector('button.copybtn');
            if (!button || button.contains(event.target)) return;
            button.click();
        });
    });
});
