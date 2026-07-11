// Custom mermaid initialization for MkDocs Material
document$.subscribe(function() {
  mermaid.initialize({ startOnLoad: false, securityLevel: 'loose' });
  
  var pres = document.querySelectorAll('pre.mermaid');
  pres.forEach(function(pre) {
    var code = pre.querySelector('code');
    if (!code) return;
    var text = code.textContent;
    var id = 'mermaid-' + Math.random().toString(36).substring(2, 8);
    
    mermaid.render(id, text).then(function(result) {
      pre.innerHTML = result.svg;
    }).catch(function(err) {
      console.warn('Mermaid render error:', err);
    });
  });
});
