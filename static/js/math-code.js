document.querySelectorAll(":not(pre) > code:not(.nolatex)").forEach((t => {
    if (t.childElementCount > 0) return;
    let e = t.textContent;
    /^\$[^$]/.test(e) && /[^$]\$$/.test(e) && (e = e.replace(/^\$/, "\\(").replace(/\$$/, "\\)"), t.textContent = e), (/^\\\((.|\s)+\\\)$/.test(e) || /^\\\[(.|\s)+\\\]$/.test(e) || /^\$(.|\s)+\$$/.test(e) || /^\\begin\{([^}]+)\}(.|\s)+\\end\{[^}]+\}$/.test(e)) && (t.outerHTML = t.innerHTML)
}));