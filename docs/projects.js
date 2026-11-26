/**
 * YBV Studio - Modular Desktop Software Registry
 *
 * Add any future Windows desktop app by appending a single object
 * to `window.YBV_SOFTWARE`.
 */
window.YBV_SOFTWARE = [
  {
    id: 'orpheus',
    icon: 'lyre',
    name: 'Orpheus',
    version: 'v1.0.11',
    status: 'active',
    statusBadges: [{ label: 'Free Download', variant: 'active' }],
    featured: true,
    category: 'Pro Video Editor with Built-In AI',
    installerFilename: 'Orpheus-v1.0.11-Setup.exe',
    sha256: '86f8abef6c8cdbdc23c40e0bb96019b6ce72a24eebc2023aae641e8de928938f',
    releaseTagUrl: 'https://github.com/hertzwaG/Orpheus/releases/tag/v1.0.11',
    fileMetaLine: 'Orpheus-v1.0.11-Setup.exe • 39 MB • Windows 10/11 64-bit • Free',
    summaryHtml:
      'Turn raw clips into <strong class="hl-blue">Shorts</strong> or full videos in minutes. Everything runs on your PC with <strong class="hl-white">no accounts or limits</strong>.',
    description:
      'Use Creator Mode for AI-assisted vertical edits and local subtitles, or switch to Studio Mode for full multi-track editing, color grading, and smart automation.',
    badges: [
      { label: 'Local AI Subtitles' },
      { label: 'Smart 9:16 Framing' },
      { label: 'AI Silence Removal' },
      { label: 'No Watermarks' }
    ],
    downloadUrl:
      'https://github.com/hertzwaG/Orpheus/releases/download/v1.0.11/Orpheus-v1.0.11-Setup.exe',
    downloadLabel: 'Download for Windows',
    downloadCtaVersion: 'Download Orpheus v1.0.11',
    sourceUrl: 'https://github.com/hertzwaG/Orpheus',
    sourceLabel: 'GitHub',
    releasesUrl: 'https://github.com/hertzwaG/Orpheus/releases',
    showSpecsButton: true,
    installSteps: [
      {
        step: '1',
        title: 'Run Orpheus-v1.0.11-Setup.exe',
        detail: 'Open the 39 MB installer and finish setup in a few seconds.'
      },
      {
        step: '2',
        title: 'Choose Creator or Studio Mode',
        detail: 'Pick Creator Mode for fast AI edits, or Studio Mode for full timeline control.'
      },
      {
        step: '3',
        title: 'Drop your video & export',
        detail: 'Generate local AI subtitles, trim silences, and save a clean MP4 with no watermark.'
      }
    ],
    smartScreenNote:
      'Windows SmartScreen tip: Because Orpheus is an independent free desktop app, Windows may show a blue prompt on first run. Click "More info" and then "Run anyway" to install.',
    specs: [
      { label: 'Version', value: 'v1.0.11' },
      { label: 'Installer', value: 'Orpheus-v1.0.11-Setup.exe (39 MB)' },
      { label: 'SHA-256', value: '86f8abef6c8cdbdc23c40e0bb96019b6ce72a24eebc2023aae641e8de928938f' },
      { label: 'System', value: 'Windows 10 & Windows 11 (64-bit)' },
      { label: 'Local AI Subtitles', value: 'Whisper model downloaded once on first use, then 100% offline' },
      { label: 'Effects & Transitions', value: 'Live color grading, transitions, and visual looks' },
      { label: 'Fast MP4 Export', value: 'Smooth, high-quality MP4 video output' },
      { label: 'Price', value: '100% Free Windows software (no subscriptions)' }
    ],
    modes: [
      {
        id: 'creator',
        theme: 'creator',
        name: 'Creator Mode',
        shortLabel: 'Creator Mode',
        badge: 'Creator Mode · AI-Assisted',
        headline: 'Fast edits powered by built-in AI.',
        summary: 'Turn raw footage into ready-to-post clips in seconds.',
        highlights: [
          {
            title: 'AI Auto-Reframe',
            detailHtml:
              'Automatically converts widescreen videos into <strong class="hl-blue">vertical 9:16 format</strong>.'
          },
          {
            title: 'Local AI Subtitles',
            detailHtml:
              'Generates accurate captions <strong class="hl-white">offline with zero cloud limits</strong>.'
          },
          {
            title: 'AI Prompt Editing',
            detailHtml:
              'Type what you want and let the AI <strong class="hl-blue">cut silences, add hooks, and style clips</strong>.'
          }
        ],
        preview: {
          type: 'creator-ai',
          headerTitle: 'AI Workspace',
          headerStatus: 'Offline AI Ready',
          bannerText: 'Smart 9:16 Auto-Framing + Live AI Captions',
          prompts: [
            { label: 'Prompt 1', text: 'Create vertical short with bold captions' },
            { label: 'Prompt 2', text: 'Remove dead air and add title hook' }
          ]
        }
      },
      {
        id: 'studio',
        theme: 'studio',
        name: 'Studio Mode',
        shortLabel: 'Studio Mode',
        badge: 'Studio Mode · Pro Timeline',
        headline: 'Full timeline control with AI tools.',
        summary: 'Multi-track editing, color grading, and smart automation in one workspace.',
        highlights: [
          {
            title: 'Multi-Track Timeline',
            detailHtml:
              'Layer <strong class="hl-purple">video, audio, and captions</strong> with smooth real-time playback.'
          },
          {
            title: 'AI Audio & Pacing',
            detailHtml:
              'Detect and <strong class="hl-white">cut silent pauses automatically</strong> with one click.'
          },
          {
            title: 'Color & Effects',
            detailHtml:
              'Grade colors live and apply <strong class="hl-purple">transitions or visual looks</strong> fast.'
          }
        ],
        preview: {
          type: 'studio-timeline',
          headerTitle: 'Multi-Track Timeline',
          headerStatus: 'AI Tools Active',
          tracks: [
            { id: 'V3', kind: 'caption', label: 'AI Generated Subtitles', span: [12, 74] },
            { id: 'V2', kind: 'fx', label: 'Color Grade & Effects', span: [6, 86] },
            { id: 'V1', kind: 'video', label: 'Main Video Track', span: [0, 95] },
            { id: 'A1', kind: 'audio', label: 'Audio & Voiceover', span: [0, 91] }
          ]
        }
      }
    ]
  },
  {
    id: 'apollo',
    icon: 'apollo',
    name: 'Apollo',
    version: null,
    status: 'coming-soon',
    statusBadges: [
      { label: 'Coming Soon', variant: 'soon' },
      { label: 'Free', variant: 'free' }
    ],
    featured: false,
    category: 'Pro Photo & Image Editor with Built-In AI',
    summaryHtml:
      'Multi-layer photo editing, retouching, and color tools enhanced with <strong class="hl-amber">local AI</strong> — fast, offline, and <strong class="hl-white">subscription-free</strong>.',
    badges: [
      { label: 'AI Photo Tools' },
      { label: 'Layers & Masks' },
      { label: '100% Offline' }
    ],
    downloadUrl: null,
    sourceUrl: 'https://github.com/hertzwaG',
    sourceLabel: 'Follow Updates on GitHub',
    showSpecsButton: false
  }
];
