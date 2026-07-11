// Mermaid initialization — fixes <pre class="mermaid"><code> nesting issue
(function initMermaid() {
  if (typeof mermaid === 'undefined') {
    setTimeout(initMermaid, 200);
    return;
  }
  mermaid.initialize({ startOnLoad: false, securityLevel: 'loose' });
  
  var pres = document.querySelectorAll('pre.mermaid');
  pres.forEach(function(pre) {
    var code = pre.querySelector('code');
    if (!code) return;
    var id = 'mermaid-' + Math.random().toString(36).substr(2, 8);
    try {
      mermaid.render(id, code.textContent).then(function(result) {
        pre.innerHTML = result.svg;
      }).catch(function(e) {
        console.warn('mermaid error:', e);
      });
    } catch(e) {
      console.warn('mermaid render exception:', e);
    }
  });
})();
