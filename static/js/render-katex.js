(e => {
    function t(t, r = e) {
        return r.querySelector(t)
    }

    function r(t, r = e) {
        return r.querySelectorAll(t)
    }

    renderMathInElement(e.body, {
        strict: !1,
        throwOnError: !1,
        trust: e => ["\\href", "\\htmlId", "\\htmlClass"].includes(e.command),
        macros: {
            "\\eqref": "\\htmlClass{ktx-eqref}{\\href{###1}{(\\text{#1})}}",
            "\\ref": "\\htmlClass{ktx-ref}{\\href{###1}{\\text{#1}}}",
            "\\label": "\\htmlClass{ktx-label}{\\htmlId{#1}{}}"
        }
    });
    const n = r(".katex:has(:is(.ktx-ref, .ktx-eqref))");
    if (!n.length) return;
    const s = {};
    let l = 0;
    r(".katex-display").forEach((e => {
        const n = ".ktx-label > [id]", a = t(n, e);
        if (!a) return;
        const o = [...r(":scope > [style]", a.offsetParent.parentNode)].map((e => t(n, e)?.id || "")),
            c = [...r(".tag > .vlist-t > .vlist-r > .vlist > [style]", e)].map((e => t(".eqn-num", e) ? `(${++l})` : e.innerText.trim()));
        o.length === c.length ? o.forEach(((e, t) => e && (s[e] = c[t]))) : console.warn("labels and tags differ in length in ", e, o, c)
    })), n.forEach((e => {
        const r = t("a", e);
        if (!r) return;
        const n = r.parentNode.classList.contains("ktx-eqref"), l = e.parentNode;
        l.after(r), l.remove(), function (e, t) {
            const r = e.getAttribute("href").replace(/^#/, ""), n = s[r];
            e.innerText = n ? t ? n : n.replace(/^\(|\)$/g, "") : "???", e.classList.add("cross-ref-eq" + (t ? "n" : ""))
        }(r, n)
    }))
})(document);