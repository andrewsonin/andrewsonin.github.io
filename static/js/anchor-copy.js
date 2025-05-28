document.addEventListener("DOMContentLoaded", function () {
    const headers = document.querySelectorAll("h1[id],h2[id],h3[id],h4[id],h5[id],h6[id]");

    headers.forEach(header => {
        const id = header.id;
        if (!id) return;

        const link = document.createElement("a");
        link.href = `#${id}`;
        link.className = "heading-anchor";
        link.innerText = "🔗";
        link.title = "Copy link";
        link.addEventListener("click", function (e) {
            e.preventDefault();
            const url = `${window.location.origin}${window.location.pathname}#${id}`;
            navigator.clipboard.writeText(url).then(() => {
                link.innerText = "✅";
                setTimeout(() => {
                    link.innerText = "🔗";
                }, 1000);
            });
        });

        header.appendChild(link);
    });
});