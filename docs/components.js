/**
 * Niru ML - Shared Components
 * Injects consistent header and footer across all pages.
 * No framework dependency. Easy to replace with template partials or components later.
 */
(function () {
    'use strict';

    var currentPage = window.location.pathname.split('/').pop() || 'index.html';

    function navLink(href, label) {
        var isActive = currentPage === href;
        return '<a href="' + href + '" class="' + (isActive ? 'active' : '') + '">' + label + '</a>';
    }

    // --- Header ---
    var headerEl = document.getElementById('site-header');
    if (headerEl) {
        headerEl.innerHTML =
            '<header class="site-header">' +
            '  <div>' +
            '    <a href="index.html">' +
            '      <img src="NIRUML.png" alt="Niru ML Logo" class="logo-img"' +
            '           onerror="this.style.display=\'none\'; this.parentElement.innerHTML=\'<span style=\\\'font-size:1.25rem;font-weight:700\\\'>NIRU ML</span>\'">' +
            '    </a>' +
            '  </div>' +
            '  <nav class="site-nav">' +
            '    ' + navLink('index.html', 'Home') +
            '    ' + navLink('about.html', 'About') +
            '    ' + navLink('pre-register.html', 'Pre-Register') +
            '  </nav>' +
            '</header>';
    }

    // --- Footer ---
    var footerEl = document.getElementById('site-footer');
    if (footerEl) {
        footerEl.innerHTML =
            '<footer class="site-footer">' +
            '  <div class="footer-links">' +
            '    <a href="mailto:brendonrt@niru.ml">brendonrt@niru.ml</a> |' +
            '    <a href="https://www.linkedin.com/in/brendon-rt/" target="_blank" rel="noopener noreferrer">Founder LinkedIn</a> |' +
            '    <a href="https://www.linkedin.com/company/niru-ml/" target="_blank" rel="noopener noreferrer">Company LinkedIn</a>' +
            '  </div>' +
            '  <div>' +
            '    <a href="https://www.linkedin.com/company/niru-ml/" target="_blank" rel="noopener noreferrer" class="social-icon">in</a>' +
            '  </div>' +
            '  <p class="copyright">&copy; 2026 Niru ML. All rights reserved.</p>' +
            '</footer>';
    }
})();
