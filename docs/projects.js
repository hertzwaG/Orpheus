/**
 * YBV Studio - Modular Desktop Software Registry
 *
 * Add any future Windows desktop app by appending a single object
 * to `window.YBV_SOFTWARE`.
 */
window.YBV_SOFTWARE = [
  {
    id: 'orpheus',
    name: 'Orpheus',
    version: 'v1.0.10',
    status: 'active',
    statusLabel: 'Free Download',
    featured: true,
    category: 'Windows Video Editor (64-bit)',
    installerFilename: 'Orpheus-v1.0.10-Setup.exe',
    fileMetaLine: 'Orpheus-v1.0.10-Setup.exe • 39 MB • Windows 10/11 64-bit • Free',
    summaryHtml:
      'Turn raw clips into <strong class="hl-blue">Shorts</strong> or full videos in minutes. Everything runs on your PC with <strong class="hl-white">no accounts or limits</strong>.',
    description:
      'Use Creator Mode for vertical 9:16 clips and auto subtitles, or switch to Studio Mode for multi-track editing, effects, and color grading.',
    badges: [
      { label: 'Auto Subtitles' },
      { label: 'Vertical 9:16 Layout' },
      { label: 'Fast MP4 Export' },
      { label: 'No Watermarks' }
    ],
    downloadUrl:
      'https://github.com/hertzwaG/Orpheus/releases/download/v1.0.10/Orpheus-v1.0.10-Setup.exe',
    downloadLabel: 'Download for Windows',
    downloadCtaVersion: 'Download Orpheus v1.0.10',
    sourceUrl: 'https://github.com/hertzwaG/Orpheus',
    releasesUrl: 'https://github.com/hertzwaG/Orpheus/releases',
    showSpecsButton: true,
    installSteps: [
      {
        step: '1',
        title: 'Run Orpheus-v1.0.10-Setup.exe',
        detail: 'Open the 39 MB installer and finish setup in a few seconds.'
      },
      {
        step: '2',
        title: 'Choose Creator or Studio Mode',
        detail: 'Pick Creator Mode for fast 9:16 Shorts, or Studio Mode for a multi-track timeline.'
      },
      {
        step: '3',
        title: 'Drop your video & export',
        detail: 'Add auto subtitles, cut your clips, and save a clean MP4 with no watermark.'
      }
    ],
    smartScreenNote:
      'Windows SmartScreen tip: Because Orpheus is an independent free app, Windows may show a blue prompt on first run. Click "More info" and then "Run anyway" to install.',
    specs: [
      { label: 'Version', value: 'v1.0.10' },
      { label: 'Installer', value: 'Orpheus-v1.0.10-Setup.exe (39 MB)' },
      { label: 'System', value: 'Windows 10 & Windows 11 (64-bit)' },
      { label: 'Auto Subtitles', value: 'English & Bulgarian (no internet required)' },
      { label: 'Effects & Transitions', value: 'Ready-made effects, filters, and transitions' },
      { label: 'Fast MP4 Export', value: 'Smooth, high-quality MP4 video output' },
      { label: 'License', value: '100% Free & Open Source' }
    ],
    modes: [
      {
        id: 'creator',
        theme: 'creator',
        name: 'Creator Mode',
        shortLabel: 'Creator Mode',
        badge: 'Creator Mode • 9:16 Shorts',
        headline: 'Vertical Shorts in one click.',
        summary: 'Made for TikTok, Reels, and YouTube Shorts.',
        highlights: [
          {
            title: 'Vertical 9:16 Layout',
            detailHtml:
              'Turn wide videos into <strong class="hl-blue">vertical 9:16 clips</strong> with your camera framed cleanly on the bottom.'
          },
          {
            title: 'Auto Subtitles',
            detailHtml:
              'Generate accurate <strong class="hl-white">English and Bulgarian captions</strong> automatically, with no internet required.'
          },
          {
            title: 'Simple Text Commands',
            detailHtml:
              'Type simple instructions like <strong class="hl-blue">\'cut silences\'</strong> or <strong class="hl-blue">\'add subtitles\'</strong> in English or Bulgarian.'
          }
        ],
        preview: {
          type: 'creator-split',
          aspectLabel: 'Vertical 9:16 Layout',
          topRegionLabel: 'Main Gameplay / Screen',
          captionSample: '“Auto subtitles placed right above FaceCam”',
          bottomRegionLabel: 'FaceCam Box',
          commandExamples: [
            { label: 'Command 1', text: 'Split to 9:16 and add captions' },
            { label: 'Command 2', text: 'Cut silences and export MP4' }
          ]
        }
      },
      {
        id: 'studio',
        theme: 'studio',
        name: 'Studio Mode',
        shortLabel: 'Studio Mode',
        badge: 'Studio Mode • Full Timeline',
        headline: 'Full control on a multi-track timeline.',
        summary: 'For longer YouTube videos, color grading, and audio mixing.',
        highlights: [
          {
            title: 'Multi-Track Timeline',
            detailHtml:
              'Layer <strong class="hl-purple">video, voiceovers, music, and captions</strong> with smooth playback.'
          },
          {
            title: 'Real-Time Color',
            detailHtml:
              'Tweak <strong class="hl-white">exposure, contrast, and color looks</strong> directly in the preview.'
          },
          {
            title: 'Effects & Transitions',
            detailHtml:
              'Apply <strong class="hl-purple">ready-made effects, filters, and transitions</strong> to your clips.'
          }
        ],
        preview: {
          type: 'studio-timeline',
          timecode: '00:04:18:12',
          tracks: [
            { id: 'V3', kind: 'caption', label: 'Auto Subtitles · EN / BG', span: [12, 70] },
            { id: 'V2', kind: 'fx', label: 'Color Look · Cinematic Grade', span: [6, 86] },
            { id: 'V1', kind: 'video', label: 'Main_Video_1080p60.mp4', span: [0, 94] },
            { id: 'A1', kind: 'audio', label: 'Voiceover_Mic.wav', span: [0, 90] }
          ]
        }
      }
    ]
  },
  {
    id: 'coming-soon',
    name: 'More Apps Coming Soon',
    version: null,
    status: 'in-development',
    statusLabel: 'In Development',
    featured: false,
    category: 'New Windows tools in development',
    summaryHtml:
      'Lightweight <strong class="hl-white">Windows desktop tools</strong> built for speed, simplicity, and privacy. More releases coming soon.',
    badges: [],
    downloadUrl: null,
    sourceUrl: 'https://github.com/hertzwaG',
    sourceLabel: 'Follow on GitHub',
    showSpecsButton: false
  }
];
