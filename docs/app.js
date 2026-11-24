(function () {
  'use strict';

  const softwareList = Array.isArray(window.YBV_SOFTWARE) ? window.YBV_SOFTWARE : [];
  if (softwareList.length === 0) return;

  const ICONS = {
    download:
      '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M8 2.25v7.5m0 0L4.75 6.5M8 9.75l3.25-3.25M2.5 12.75h11"/></svg>',
    github:
      '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6.2 12.6c-2.8.85-2.8-1.4-3.9-1.7m7.8 3.4v-2.2a1.9 1.9 0 0 0-.55-1.5c1.8-.2 3.7-.9 3.7-4a3.1 3.1 0 0 0-.82-2.15 2.9 2.9 0 0 0-.08-2.12s-.67-.21-2.2.82a7.6 7.6 0 0 0-4 0C4.62 2.12 3.95 2.33 3.95 2.33a2.9 2.9 0 0 0-.08 2.12 3.1 3.1 0 0 0-.82 2.15c0 3.1 1.9 3.8 3.7 4a1.9 1.9 0 0 0-.55 1.5v2.2"/></svg>',
    specs:
      '<svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="2.25" y="2.75" width="11.5" height="10.5" rx="2"/><path d="M5.25 6h5.5M5.25 9.5h3.5"/></svg>'
  };

  function escapeHtml(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function renderYbvBadge(sizeClass) {
    const cls = sizeClass ? 'ybv-badge ' + sizeClass : 'ybv-badge';
    return '<span class="' + cls + '" translate="no">YBV</span>';
  }

  function renderBadges(badges) {
    if (!Array.isArray(badges) || badges.length === 0) return '';
    return (
      '<ul class="badge-list" aria-label="Highlights">' +
      badges
        .map(function (item) {
          const label = typeof item === 'string' ? item : item.label;
          return '<li class="badge-chip">' + escapeHtml(label) + '</li>';
        })
        .join('') +
      '</ul>'
    );
  }

  function renderStatusPill(item) {
    const isActive = item.status === 'active';
    const cls = isActive ? 'status-pill status-pill--active' : 'status-pill status-pill--dev';
    return (
      '<span class="' +
      cls +
      '"><span class="status-pill__dot" aria-hidden="true"></span>' +
      escapeHtml(item.statusLabel || (isActive ? 'Free Download' : 'In Development')) +
      '</span>'
    );
  }

  function renderModePreview(mode) {
    const preview = mode.preview;
    if (!preview) return '';

    if (preview.type === 'creator-split') {
      const commandsHtml = (preview.commandExamples || [])
        .map(function (cmd) {
          return (
            '<div class="command-card">' +
            '<span class="command-card__lang">' +
            escapeHtml(cmd.label || cmd.lang || 'Text command') +
            '</span>' +
            '<span class="command-card__text">“' +
            escapeHtml(cmd.text) +
            '”</span>' +
            '</div>'
          );
        })
        .join('');

      return (
        '<div class="workspace-stage" aria-label="Creator Mode layout preview">' +
        '<div class="stage-header">' +
        '<span class="stage-header__title">' +
        escapeHtml(preview.aspectLabel) +
        '</span>' +
        '<span class="mono">Auto Subtitles</span>' +
        '</div>' +
        '<div class="creator-diagram">' +
        '<div class="vertical-frame" role="img" aria-label="9 by 16 vertical layout showing main screen on top, auto subtitles in the middle, and FaceCam on the bottom">' +
        '<div class="vf-zone vf-zone--capture">' +
        escapeHtml(preview.topRegionLabel) +
        '</div>' +
        '<div class="vf-zone vf-zone--subtitles">' +
        escapeHtml(preview.captionSample) +
        '</div>' +
        '<div class="vf-zone vf-zone--facecam">' +
        escapeHtml(preview.bottomRegionLabel) +
        '</div>' +
        '</div>' +
        '<div class="command-feed">' +
        '<span class="command-feed__label">Simple text commands (English or Bulgarian)</span>' +
        commandsHtml +
        '</div>' +
        '</div>' +
        '</div>'
      );
    }

    if (preview.type === 'studio-timeline') {
      const tracksHtml = (preview.tracks || [])
        .map(function (track) {
          const left = Number(track.span[0]) || 0;
          const width = Math.max(8, (Number(track.span[1]) || 20) - left);
          return (
            '<div class="track-row">' +
            '<span class="track-id">' +
            escapeHtml(track.id) +
            '</span>' +
            '<div class="track-lane">' +
            '<span class="track-clip track-clip--' +
            escapeHtml(track.kind) +
            '" style="left:' +
            left +
            '%;width:' +
            width +
            '%;">' +
            escapeHtml(track.label) +
            '</span>' +
            '</div>' +
            '</div>'
          );
        })
        .join('');

      return (
        '<div class="workspace-stage" aria-label="Studio Mode multi-track timeline preview">' +
        '<div class="stage-header">' +
        '<span class="stage-header__title">Multi-Track Timeline &amp; Color</span>' +
        '<span class="mono">' +
        escapeHtml(preview.timecode) +
        '</span>' +
        '</div>' +
        '<div class="studio-timeline">' +
        '<div class="timeline-ruler" aria-hidden="true">' +
        '<span>00:00</span><span>00:02</span><span>00:04</span><span>00:06</span><span>00:08</span>' +
        '</div>' +
        tracksHtml +
        '</div>' +
        '</div>'
      );
    }

    return '';
  }

  function renderHeroPreview(flagship) {
    const container = document.getElementById('hero-app-preview');
    if (!container || !flagship) return;

    const modes = Array.isArray(flagship.modes) ? flagship.modes : [];
    const params = new URLSearchParams(window.location.search);
    const requestedMode = params.get('mode');
    const initialMode =
      modes.find(function (m) {
        return m.id === requestedMode;
      }) || modes[0];

    const tabsHtml = modes
      .map(function (mode) {
        const isSelected = initialMode && mode.id === initialMode.id;
        return (
          '<button type="button" class="mode-tab" role="tab" ' +
          'id="tab-' +
          escapeHtml(mode.id) +
          '" ' +
          'aria-controls="panel-' +
          escapeHtml(mode.id) +
          '" ' +
          'aria-selected="' +
          (isSelected ? 'true' : 'false') +
          '" ' +
          'tabindex="' +
          (isSelected ? '0' : '-1') +
          '" ' +
          'data-mode-id="' +
          escapeHtml(mode.id) +
          '">' +
          escapeHtml(mode.shortLabel || mode.name) +
          '</button>'
        );
      })
      .join('');

    const panelsHtml = modes
      .map(function (mode) {
        const isSelected = initialMode && mode.id === initialMode.id;
        const badgeClass =
          mode.theme === 'studio'
            ? 'mode-copy__badge mode-copy__badge--studio'
            : 'mode-copy__badge mode-copy__badge--creator';

        const highlightsHtml = (mode.highlights || [])
          .map(function (item) {
            const detailContent = item.detailHtml ? item.detailHtml : escapeHtml(item.detail);
            return (
              '<li class="feature-item">' +
              '<h3 class="feature-item__title">' +
              escapeHtml(item.title) +
              '</h3>' +
              '<p class="feature-item__detail">' +
              detailContent +
              '</p>' +
              '</li>'
            );
          })
          .join('');

        return (
          '<div class="mode-panel" role="tabpanel" ' +
          'id="panel-' +
          escapeHtml(mode.id) +
          '" ' +
          'aria-labelledby="tab-' +
          escapeHtml(mode.id) +
          '" ' +
          (isSelected ? '' : 'hidden') +
          '>' +
          '<div class="mode-copy">' +
          (mode.badge
            ? '<span class="' + badgeClass + '">' + escapeHtml(mode.badge) + '</span>'
            : '') +
          '<h2 class="mode-copy__heading">' +
          escapeHtml(mode.headline) +
          '</h2>' +
          '<p class="mode-copy__summary">' +
          escapeHtml(mode.summary) +
          '</p>' +
          '<ul class="feature-list">' +
          highlightsHtml +
          '</ul>' +
          '</div>' +
          renderModePreview(mode) +
          '</div>'
        );
      })
      .join('');

    const activeModeId = initialMode ? initialMode.id : 'creator';

    container.innerHTML =
      '<div class="app-window" data-active-mode="' +
      escapeHtml(activeModeId) +
      '" aria-label="' +
      escapeHtml(flagship.name) +
      ' interactive workspace preview">' +
      '<div class="app-window__titlebar">' +
      '<div class="app-window__identity">' +
      renderYbvBadge('ybv-badge--sm') +
      '<span class="app-window__title" translate="no">' +
      escapeHtml(flagship.name) +
      '</span>' +
      '<span class="version-tag">' +
      escapeHtml(flagship.version) +
      '</span>' +
      renderStatusPill(flagship) +
      '</div>' +
      '<div class="mode-tablist" role="tablist" aria-label="' +
      escapeHtml(flagship.name) +
      ' workspace modes">' +
      tabsHtml +
      '</div>' +
      '</div>' +
      panelsHtml +
      '<div class="app-window__footer">' +
      renderBadges(flagship.badges) +
      '<span class="mono" style="font-size:0.8rem;color:var(--ink-bright-body);">' +
      escapeHtml(flagship.fileMetaLine || '') +
      '</span>' +
      '</div>' +
      '</div>';

    bindModeTabs(container);
  }

  function bindModeTabs(root) {
    const appWindow = root.querySelector('.app-window');
    const tabs = Array.prototype.slice.call(root.querySelectorAll('[role="tab"]'));
    if (tabs.length === 0) return;

    function activateTab(nextTab, moveFocus) {
      const targetId = nextTab.getAttribute('data-mode-id');
      if (appWindow && targetId) {
        appWindow.setAttribute('data-active-mode', targetId);
      }

      tabs.forEach(function (tab) {
        const isCurrent = tab === nextTab;
        tab.setAttribute('aria-selected', isCurrent ? 'true' : 'false');
        tab.setAttribute('tabindex', isCurrent ? '0' : '-1');
        const panelId = tab.getAttribute('aria-controls');
        const panel = panelId ? document.getElementById(panelId) : null;
        if (panel) {
          panel.hidden = !isCurrent;
        }
      });

      if (moveFocus) {
        nextTab.focus();
      }

      if (targetId && window.history && window.history.replaceState) {
        try {
          const url = new URL(window.location.href);
          url.searchParams.set('mode', targetId);
          window.history.replaceState(null, '', url.toString());
        } catch (_) {
          // Ignore file:// URL restrictions
        }
      }
    }

    tabs.forEach(function (tab, index) {
      tab.addEventListener('click', function () {
        activateTab(tab, false);
      });

      tab.addEventListener('keydown', function (event) {
        let nextIndex = null;
        if (event.key === 'ArrowRight') nextIndex = (index + 1) % tabs.length;
        else if (event.key === 'ArrowLeft') nextIndex = (index - 1 + tabs.length) % tabs.length;
        else if (event.key === 'Home') nextIndex = 0;
        else if (event.key === 'End') nextIndex = tabs.length - 1;

        if (nextIndex !== null) {
          event.preventDefault();
          activateTab(tabs[nextIndex], true);
        }
      });
    });
  }

  function renderCatalog(items) {
    const grid = document.getElementById('software-catalog-grid');
    if (!grid) return;

    grid.innerHTML = items
      .map(function (item) {
        const isActive = item.status === 'active';
        const cardClass = isActive ? 'software-card' : 'software-card software-card--dev';
        const summaryContent = item.summaryHtml ? item.summaryHtml : escapeHtml(item.summary);

        const primaryAction = item.downloadUrl
          ? '<a class="btn btn--primary" href="' +
            escapeHtml(item.downloadUrl) +
            '" data-download-app="' +
            escapeHtml(item.id) +
            '">' +
            ICONS.download +
            '<span>' +
            escapeHtml(item.downloadCtaVersion || item.downloadLabel) +
            '</span></a>'
          : '';

        const sourceAction = item.sourceUrl
          ? '<a class="btn btn--secondary" href="' +
            escapeHtml(item.sourceUrl) +
            '" target="_blank" rel="noopener noreferrer">' +
            ICONS.github +
            '<span>' +
            escapeHtml(item.sourceLabel || 'GitHub') +
            '</span></a>'
          : '';

        const specsAction = item.showSpecsButton
          ? '<button type="button" class="btn btn--ghost" data-open-specs="' +
            escapeHtml(item.id) +
            '">' +
            ICONS.specs +
            '<span>Specs</span></button>'
          : '';

        return (
          '<article class="' +
          cardClass +
          '">' +
          '<div>' +
          '<div class="software-card__head">' +
          '<div class="software-card__title-group">' +
          renderYbvBadge('ybv-badge--sm') +
          '<div>' +
          '<h3 class="software-card__title">' +
          escapeHtml(item.name) +
          '</h3>' +
          '<p class="software-card__meta">' +
          escapeHtml(item.category) +
          '</p>' +
          '</div>' +
          '</div>' +
          renderStatusPill(item) +
          '</div>' +
          '<p class="software-card__summary">' +
          summaryContent +
          '</p>' +
          renderBadges(item.badges) +
          '</div>' +
          '<div class="software-card__footer">' +
          primaryAction +
          sourceAction +
          specsAction +
          '</div>' +
          '</article>'
        );
      })
      .join('');
  }

  function setupDialogs(items) {
    const byId = new Map();
    items.forEach(function (item) {
      byId.set(item.id, item);
    });

    const specDialog = document.getElementById('software-spec-dialog');
    const specTitleEl = document.getElementById('spec-dialog-title');
    const specSubtitleEl = document.getElementById('spec-dialog-subtitle');
    const specDescEl = document.getElementById('spec-dialog-desc');
    const specTbodyEl = document.getElementById('spec-dialog-rows');
    const specFooterEl = document.getElementById('spec-dialog-footer');
    const specCloseBtn = document.getElementById('spec-dialog-close');

    const dlDialog = document.getElementById('download-helper-dialog');
    const dlTitleEl = document.getElementById('dl-dialog-title');
    const dlSubtitleEl = document.getElementById('dl-dialog-subtitle');
    const dlDirectLinkEl = document.getElementById('dl-direct-link');
    const dlReleasesLinkEl = document.getElementById('dl-releases-link');
    const dlStepsEl = document.getElementById('dl-steps-list');
    const dlSmartScreenEl = document.getElementById('dl-smartscreen-note');
    const dlCloseBtn = document.getElementById('dl-dialog-close');
    const dlDoneBtn = document.getElementById('dl-dialog-done');

    let lastTrigger = null;

    function openDialogElement(dialogEl) {
      if (!dialogEl) return;
      if (typeof dialogEl.showModal === 'function') {
        if (!dialogEl.open) dialogEl.showModal();
      } else {
        dialogEl.setAttribute('open', 'open');
      }
    }

    function closeDialogElement(dialogEl) {
      if (!dialogEl) return;
      if (typeof dialogEl.close === 'function' && dialogEl.open) {
        dialogEl.close();
      } else {
        dialogEl.removeAttribute('open');
      }
      if (lastTrigger && typeof lastTrigger.focus === 'function') {
        lastTrigger.focus();
      }
    }

    function openSpecsFor(id, triggerEl) {
      const item = byId.get(id);
      if (!item || !specDialog) return;

      lastTrigger = triggerEl || null;
      specTitleEl.textContent = item.name + (item.version ? ' ' + item.version : '');
      specSubtitleEl.textContent = item.category;
      specDescEl.textContent = item.description || '';

      specTbodyEl.innerHTML = (item.specs || [])
        .map(function (row) {
          return (
            '<tr><th scope="row">' +
            escapeHtml(row.label) +
            '</th><td>' +
            escapeHtml(row.value) +
            '</td></tr>'
          );
        })
        .join('');

      let footerHtml = '';
      if (item.downloadUrl) {
        footerHtml +=
          '<a class="btn btn--primary" href="' +
          escapeHtml(item.downloadUrl) +
          '" data-download-app="' +
          escapeHtml(item.id) +
          '">' +
          ICONS.download +
          '<span>' +
          escapeHtml(item.downloadCtaVersion || item.downloadLabel) +
          '</span></a>';
      }
      if (item.sourceUrl) {
        footerHtml +=
          '<a class="btn btn--secondary" href="' +
          escapeHtml(item.sourceUrl) +
          '" target="_blank" rel="noopener noreferrer">' +
          ICONS.github +
          '<span>GitHub</span></a>';
      }
      specFooterEl.innerHTML = footerHtml;

      openDialogElement(specDialog);
    }

    function openDownloadHelperFor(id, triggerEl) {
      const item = byId.get(id) || items[0];
      if (!item || !dlDialog) return;

      if (specDialog && specDialog.open) {
        closeDialogElement(specDialog);
      }

      lastTrigger = triggerEl || null;
      dlTitleEl.textContent = 'Downloading ' + item.name + ' ' + item.version + '…';
      dlSubtitleEl.textContent = item.fileMetaLine || item.installerFilename || '';

      if (dlDirectLinkEl && item.downloadUrl) {
        dlDirectLinkEl.setAttribute('href', item.downloadUrl);
        dlDirectLinkEl.textContent = item.installerFilename || 'Direct download link';
      }
      if (dlReleasesLinkEl && item.releasesUrl) {
        dlReleasesLinkEl.setAttribute('href', item.releasesUrl);
      }

      if (dlStepsEl && Array.isArray(item.installSteps)) {
        dlStepsEl.innerHTML = item.installSteps
          .map(function (s) {
            return (
              '<li class="dialog-step">' +
              '<span class="dialog-step__num">' +
              escapeHtml(s.step) +
              '</span>' +
              '<div>' +
              '<h3 class="dialog-step__title">' +
              escapeHtml(s.title) +
              '</h3>' +
              '<p class="dialog-step__detail">' +
              escapeHtml(s.detail) +
              '</p>' +
              '</div>' +
              '</li>'
            );
          })
          .join('');
      }

      if (dlSmartScreenEl) {
        dlSmartScreenEl.textContent = item.smartScreenNote || '';
      }

      openDialogElement(dlDialog);
    }

    document.addEventListener('click', function (event) {
      const dlTrigger = event.target.closest('[data-download-app]');
      if (dlTrigger) {
        const appId = dlTrigger.getAttribute('data-download-app') || 'orpheus';
        openDownloadHelperFor(appId, dlTrigger);
        return;
      }

      const specTrigger = event.target.closest('[data-open-specs]');
      if (specTrigger) {
        openSpecsFor(specTrigger.getAttribute('data-open-specs'), specTrigger);
      }
    });

    [
      [specDialog, specCloseBtn],
      [dlDialog, dlCloseBtn],
      [dlDialog, dlDoneBtn]
    ].forEach(function (pair) {
      const dialogEl = pair[0];
      const btnEl = pair[1];
      if (btnEl && dialogEl) {
        btnEl.addEventListener('click', function () {
          closeDialogElement(dialogEl);
        });
      }
    });

    [specDialog, dlDialog].forEach(function (dialogEl) {
      if (!dialogEl) return;
      dialogEl.addEventListener('click', function (event) {
        if (event.target === dialogEl) {
          closeDialogElement(dialogEl);
        }
      });
    });
  }

  function setupScreenshotLightbox() {
    const lightbox = document.getElementById('screenshot-lightbox');
    if (!lightbox) return;

    const imgEl = document.getElementById('lightbox-img');
    const captionEl = document.getElementById('lightbox-caption');
    const closeBtn = document.getElementById('lightbox-close');
    const prevBtn = document.getElementById('lightbox-prev');
    const nextBtn = document.getElementById('lightbox-next');

    const triggers = Array.prototype.slice.call(
      document.querySelectorAll('[data-lightbox-src]')
    );
    if (triggers.length === 0) return;

    let currentIndex = 0;
    let lastTrigger = null;

    function showSlide(index) {
      currentIndex = (index + triggers.length) % triggers.length;
      const trigger = triggers[currentIndex];
      const src = trigger.getAttribute('data-lightbox-src');
      const caption = trigger.getAttribute('data-lightbox-caption') || '';
      const alt = trigger.getAttribute('data-lightbox-alt') || caption;

      if (imgEl && src) {
        imgEl.setAttribute('src', src);
        imgEl.setAttribute('alt', alt);
      }
      if (captionEl) {
        captionEl.textContent = caption;
      }
    }

    function openLightbox(index, triggerEl) {
      lastTrigger = triggerEl || null;
      showSlide(index);
      if (typeof lightbox.showModal === 'function') {
        if (!lightbox.open) lightbox.showModal();
      } else {
        lightbox.setAttribute('open', 'open');
      }
    }

    function closeLightbox() {
      if (typeof lightbox.close === 'function' && lightbox.open) {
        lightbox.close();
      } else {
        lightbox.removeAttribute('open');
      }
      if (lastTrigger && typeof lastTrigger.focus === 'function') {
        lastTrigger.focus();
      }
    }

    triggers.forEach(function (trigger, idx) {
      trigger.addEventListener('click', function () {
        openLightbox(idx, trigger);
      });
    });

    if (closeBtn) {
      closeBtn.addEventListener('click', closeLightbox);
    }
    if (prevBtn) {
      prevBtn.addEventListener('click', function () {
        showSlide(currentIndex - 1);
      });
    }
    if (nextBtn) {
      nextBtn.addEventListener('click', function () {
        showSlide(currentIndex + 1);
      });
    }

    lightbox.addEventListener('click', function (event) {
      if (event.target === lightbox) {
        closeLightbox();
      }
    });

    lightbox.addEventListener('keydown', function (event) {
      if (event.key === 'ArrowLeft') {
        event.preventDefault();
        showSlide(currentIndex - 1);
      } else if (event.key === 'ArrowRight') {
        event.preventDefault();
        showSlide(currentIndex + 1);
      }
    });
  }

  function init() {
    const flagship =
      softwareList.find(function (item) {
        return item.featured;
      }) || softwareList[0];

    renderHeroPreview(flagship);
    renderCatalog(softwareList);
    setupDialogs(softwareList);
    setupScreenshotLightbox();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
