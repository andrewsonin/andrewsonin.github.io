(e => {
    function t(e) {
        if (1 !== e.childElementCount) return !1;
        const t = e.childNodes;
        if (1 === t.length) return !0;
        for (let e in t) {
            let n = t[e];
            if ("#text" === n.nodeName && !/^\s$/.test(n.textContent)) return !1
        }
        return !0
    }

    ["img", "embed", "object"].forEach((function (n) {
        e.querySelectorAll(n).forEach((e => {
            let r = e.parentElement;
            if (t(r)) {
                const l = "A" === r.nodeName;
                if (l) {
                    if (r = r.parentElement, !t(r)) return;
                    r.firstElementChild.style.border = "none"
                }
                "P" === r.nodeName && (r.style.textAlign = "center", l || "img" !== n || /^data:/.test(e.src) || (r.innerHTML = `<a href="${e.src}" style="border: none;">${e.outerHTML}</a>`))
            }
        }))
    })), e.querySelectorAll("p").forEach((e => {
        "* * *" === e.innerText && (e.style.textAlign = "center")
    }))
})(document);