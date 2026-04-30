document.addEventListener('DOMContentLoaded', () => {
    // Mobile menu toggle
    const menuToggle = document.getElementById('menu-toggle');
    const sidebar = document.getElementById('sidebar');

    if (menuToggle) {
        menuToggle.addEventListener('click', () => {
            sidebar.classList.toggle('active');
            menuToggle.textContent = sidebar.classList.contains('active') ? '✕' : '☰';
        });
    }

    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 768 && 
            sidebar.classList.contains('active') && 
            !sidebar.contains(e.target) && 
            e.target !== menuToggle) {
            sidebar.classList.remove('active');
            menuToggle.textContent = '☰';
        }
    });

    // ─── Collapsible navigation ───
    const navItems = sidebar.querySelectorAll('nav li');

    navItems.forEach(li => {
        const childUl = li.querySelector(':scope > ul');
        if (!childUl) return;

        const link = li.querySelector(':scope > a');
        if (!link) return;

        // Create wrapper div for the link row
        const wrapper = document.createElement('div');
        wrapper.className = 'nav-toggle';

        // Move the link inside the wrapper
        li.insertBefore(wrapper, link);
        wrapper.appendChild(link);

        // Create chevron button
        const chevron = document.createElement('span');
        chevron.className = 'chevron';
        chevron.innerHTML = '▾';
        chevron.setAttribute('aria-label', 'Toggle submenu');
        wrapper.appendChild(chevron);

        // Mark the child list as collapsible
        childUl.classList.add('collapsible');

        // Start collapsed for level-2+ (children of top-level items)
        const isTopLevel = li.parentElement === sidebar.querySelector('nav > ul');
        if (!isTopLevel) {
            childUl.classList.add('collapsed');
            wrapper.classList.add('collapsed');
        }

        // Toggle on chevron click
        chevron.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            childUl.classList.toggle('collapsed');
            wrapper.classList.toggle('collapsed');
        });

        // Also toggle when clicking the link text
        link.addEventListener('click', (e) => {
            if (childUl.classList.contains('collapsed')) {
                childUl.classList.remove('collapsed');
                wrapper.classList.remove('collapsed');
            }
        });
    });

    // ─── Smooth scrolling for anchor links ───
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                history.pushState(null, null, targetId);

                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });

                // Close mobile menu if open
                if (window.innerWidth <= 768) {
                    sidebar.classList.remove('active');
                    menuToggle.textContent = '☰';
                }
            }
        });
    });

    // ─── Highlight active section + auto-expand its parent in nav ───
    const sections = document.querySelectorAll('section');
    const navLinks = sidebar.querySelectorAll('nav a');

    function updateActiveSection() {
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            if (pageYOffset >= (sectionTop - 150)) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === '#' + current) {
                link.classList.add('active');

                // Auto-expand parent collapsibles so the active link is visible
                let parent = link.closest('.collapsible');
                while (parent) {
                    parent.classList.remove('collapsed');
                    const toggle = parent.previousElementSibling;
                    if (toggle && toggle.classList.contains('nav-toggle')) {
                        toggle.classList.remove('collapsed');
                    }
                    parent = parent.parentElement.closest('.collapsible');
                }
            }
        });
    }

    // Debounce scroll handler for performance
    let scrollTimer;
    window.addEventListener('scroll', () => {
        if (scrollTimer) cancelAnimationFrame(scrollTimer);
        scrollTimer = requestAnimationFrame(updateActiveSection);
    });

    // Run once on load to set initial state
    updateActiveSection();
});