import { IProject } from '@/types';
// import ProjectVideoCarousel from '@/components/ProjectVideoCarousel';

export const GENERAL_INFO = {
  email: 'vishalkrishnakkr@gmail.com',
  linkedin: 'https://www.linkedin.com/in/vishal-krishna-kumar-65583a201/',

  emailSubject: "Let's collaborate on a project",
  emailBody:
    'Hi Vishal,\n\nI came across your work and wanted to reach out about...',

  oldPortfolio: 'https://www.vishal-krishna.me/',
  upworkProfile: '',
};

export const SOCIAL_LINKS = [
  { name: 'GitHub', url: 'https://github.com/Vishal-Krishna-Kumar' },
  { name: 'LinkedIn', url: 'https://www.linkedin.com/in/vishal-krishna-kumar-65583a201/' },
  { name: 'Instagram', url: 'https://www.instagram.com/_vishal.27' },
  { name: 'Portfolio CLI', url: GENERAL_INFO.oldPortfolio },
];

export const MY_STACK = {
  ml_ai: [
    { name: 'Machine Learning', icon: '/logo/ml.png' },
    { name: 'Deep Learning', icon: '/logo/dl.png' },
    { name: 'NLP', icon: '/logo/nlp.png' },
    { name: 'Computer Vision', icon: '/logo/cv.png' },
    { name: 'RAG', icon: '/logo/rag.png' },
    { name: 'GraphRAG', icon: '/logo/graphrag.png' },
  ],

  frameworks: [
    { name: 'PyTorch', icon: '/logo/pytorch.png' },
    { name: 'TensorFlow', icon: '/logo/tensorflow.png' },
    { name: 'Scikit-learn', icon: '/logo/sklearn.png' },
    { name: 'OpenCV', icon: '/logo/opencv.png' },
    { name: 'Hugging Face', icon: '/logo/huggingface.png' },
    { name: 'LangChain', icon: '/logo/langchain.png' },
  ],

  database: [
    { name: 'MySQL', icon: '/logo/mysql.svg' },
    { name: 'Pinecone / FAISS', icon: '/logo/vector-db.png' },
    { name: 'PostgreSQL', icon: '/logo/postgreSQL.png' },
  ],

  tools: [
    { name: 'Git', icon: '/logo/git.png' },
    { name: 'Docker', icon: '/logo/docker.svg' },
    { name: 'AWS', icon: '/logo/aws.png' },
    { name: 'Neo4j', icon: '/logo/neo4j.png' },
  ],
};

export const PROJECTS: IProject[] = [
  {
    title: 'Graph-Augmented Intelligence',
    slug: 'GAI',
    sourceCode: 'https://github.com/Vishal-Krishna-Kumar/GraphAugmented-Intelligence_GAI',
    year: 2025,

     description: `
        Knowledge-graph retrieval-augmented generation (KG-RAG) for biomedical question answering.
        Built on the open-source <a href="https://github.com/BaranziniLab/KG_RAG" target="_blank" rel="noopener noreferrer">KG-RAG framework from the Baranzini Lab, UCSF</a>.<br/><br/>

        What I did:<br/>
        <ul>
          <li>Evaluated GPT-4 and Llama on the BiomixQA benchmark using the KG-RAG framework.</li>
        </ul><br/>

        Credit: the KG-RAG framework was created by the Baranzini Lab, UCSF. BiomixQA is an existing benchmark, not my dataset.
      `,
        role: `
      Project — applied and evaluated an existing framework
      <ul>
        <li>Evaluated GPT-4 and Llama on BiomixQA with the KG-RAG framework (Baranzini Lab, UCSF).</li>
      </ul>
      `,

    techStack: ['KG-RAG (Baranzini Lab, UCSF)', 'GPT-4', 'Llama', 'BiomixQA (evaluation benchmark)'],


    demoViedo: [
      '/projects/images/gaidemo-1.gif',
      '/projects/images/gaidemo-2.mp4',
      '/projects/images/gaidemo-3.mp4',
      '/projects/images/gaidemo-4.mp4',
    ],

    thumbnail: '/projects/thumbnail/kg-rag-1.png',
    longThumbnail: '/projects/long/kg-rag-1.png',

    images: [
      {
        src: '/projects/images/gai-1.2.png',
        title: 'KG-RAG Schematic',
        description:
          'Schematic of the KG-RAG framework (credit: Baranzini Lab, UCSF): prompt → disease-entity recognition → graph retrieval → prompt-aware context → LLM.',
      },
      {
        src: '/projects/images/gai-1.3.png',
        title: 'Example Question Source',
        description:
          'FDA news page on setmelanotide for Bardet-Biedl syndrome — the example used in the KG-RAG project README (Baranzini Lab, UCSF).',
      },
    ],
  },





























  {
    title: '3D Attention U-Net for Brain Tumor Segmentation',
    slug: '3d-unet-brain-tumor',
    sourceCode: 'https://github.com/Vishal-Krishna-Kumar/BRaTS-attention-Tumor-Segmentation-UNet-',
    year: 2025,

     description: `
        🏆 <strong>2nd Place, Hack for Research 2025</strong> <br/>

        Brain tumor segmentation on multimodal MRI from the BraTS 2019 dataset with a 3D Attention U-Net in PyTorch.
        I trained and modified public code from Islam et al. (2019)
        (<a href="https://github.com/mobarakol/3D_Attention_UNet" target="_blank" rel="noopener noreferrer">3D_Attention_UNet</a>);
        the architecture is theirs.<br/><br/>

        Key Facts:<br/>
        <ul>
            <li>🏆 2nd place, Hack for Research 2025 (team lead).</li>
            <li>🧠 Model: 3D U-Net with attention gates, from Islam et al. (2019).</li>
            <li>🧬 Data: BraTS 2019 multimodal MRI.</li>
            <li>⚙️ Framework: PyTorch.</li>
        </ul>
`,

         role: `
      Team Lead <br/>
      <ul>
        <li>✅ Led the team that placed 2nd at Hack for Research 2025.</li>
        <li>🧠 Trained and modified the public 3D Attention U-Net code from Islam et al. (2019) on BraTS 2019.</li>
      </ul>
      `,


    techStack: ['PyTorch', '3D Attention U-Net (Islam et al., 2019)', 'BraTS 2019'],

    thumbnail: '/projects/thumbnail/3d-attention-1.webp',
    longThumbnail: '/projects/long/3d-attention-1.webp',

    images: [
      {
        src: '/projects/images/3d-attention-1.1.png',
        title: 'Figure from Islam et al. (2019)',
        description:
          'BraTS 2019 examples and Dice comparison (3D Attention UNet vs 3D UNet) from the Islam et al. (2019) paper — the authors\' results, not mine.',
      },
      {
        src: '/projects/images/3d-attention-1.2.png',
        title: 'Sample Predictions',
        description:
          'T1ce / FLAIR / T2 slices with ground-truth and predicted tumor masks.',
      },
      {
        src: '/projects/images/3d-attention-1.3.png',
        title: 'Reference Architecture (third-party)',
        description:
          'Third-party figure of a 3D U-Net with CoT attention layers — shown for reference; not this project\'s model.',
      },
      {
        src: '/projects/images/3d-attention-1.4.png',
        title: '3D U-Net Concept (Çiçek et al., 2016)',
        description:
          'Figure from the original 3D U-Net paper by Çiçek et al. (2016) — credit to the authors.',
      },
    ],
  },




























  {
    title: 'Deepfake Detection System',
    slug: 'deepfake-detection',
    sourceCode: 'https://github.com/Vishal-Krishna-Kumar/DeepFake-Detection-System',
    year: 2023,

     description: `
Final-year undergraduate team project at SRM Institute of Science and Technology: a deepfake detection system using <strong>XceptionNet</strong>, built with <strong>TensorFlow/Keras</strong>.<br/><br/>

Publication:<br/>
S. Ezra Vethamani, Kawin P., Rishi R., Vishal K. "Deepfake Detection System: AI and Deep Learning Framework for Cyber Verification." TDEMAS, 2023.<br/><br/>

Image credit: the animated demo below is from the open-source <a href="https://github.com/wdrink/PyDeepFakeDet" target="_blank" rel="noopener noreferrer">PyDeepFakeDet</a> toolkit (Fudan Vision and Learning Lab), not output of this project.
`
,
role: `
      Team member — final-year undergraduate team project (SRM Institute of Science and Technology)
      <ul>
        <li>📄 Co-author of the TDEMAS 2023 paper on the project.</li>
        <li>🧠 Deepfake detection with XceptionNet in TensorFlow/Keras.</li>
      </ul>
      `,
    techStack: ['TensorFlow', 'Keras', 'XceptionNet'],

    thumbnail: '/projects/thumbnail/deepfake-1.webp',
    longThumbnail: '/projects/long/deepfake-1.webp',

    images: [
      {
        src: '/projects/images/deepfake-1.3.gif',
        title: 'Demo from PyDeepFakeDet (third-party)',
        description:
          'Credit: PyDeepFakeDet toolkit, Fudan Vision and Learning Lab — not output of this project.',
      },
    ],
  },

































  

  {
    title: 'AlphaZero-Style Game Agent',
    slug: 'alpha-zero-rl',
    sourceCode: 'https://github.com/Vishal-Krishna-Kumar/AlphaZero-Chess',
    year: 2024,

     description: `AlphaZero-style game agent: self-play reinforcement learning with Monte Carlo Tree Search (MCTS) guided by a
        policy-value network, in PyTorch with CUDA. Extended
        <a href="https://github.com/foersterrobert/AlphaZeroFromScratch" target="_blank" rel="noopener noreferrer">Robert Förster's AlphaZero tutorial</a>.
        Reached an 81% win rate vs rule-based opponents.
<br/><br/>

        Key Facts:<br/>
        <ul>
            <li>🌲 Monte Carlo Tree Search guided by a policy-value network.</li>
            <li>⚙️ PyTorch and CUDA.</li>
            <li>🏁 81% win rate vs rule-based opponents.</li>
        </ul><br/>

        Credit: the base implementation comes from Robert Förster's AlphaZero tutorial.

`,
role: `
Personal project — extended Robert Förster's AlphaZero tutorial <br/>
<ul>
  <li>🌲 Worked with MCTS and a policy-value network in PyTorch (CUDA).</li>
  <li>🏁 Reached an 81% win rate vs rule-based opponents.</li>
</ul>
      `,

    techStack: ['PyTorch', 'CUDA', 'Monte Carlo Tree Search', 'Policy-value network'],

    thumbnail: '/projects/thumbnail/alphazero-1.jpg',
    longThumbnail: '/projects/long/alphazero-1.jpg',

    images: [
      {
        src: '/projects/images/alphazero-1.1.jpg',
        title: 'Go Board',
        description:
          'Illustrative photo (third-party image).',
      },
      {
        src: '/projects/images/alphazero-1.2.png',
        title: 'Separate Project: ChessAI',
        description:
          'Screenshot of my separate Pygame ChessAI project — not the AlphaZero-style agent.',
      },
      {
        src: '/projects/images/alphazero-1.3.jpg',
        title: 'Generic CNN Diagram',
        description:
          'Illustrative convolutional-network diagram (third-party image) — not this project\'s policy-value network.',
      },
      {
        src: '/projects/images/alphazero-1.4.png',
        title: 'Othello Agent Figure (third-party)',
        description:
          'Third-party figure of a learned Othello agent — not from this project.',
      },
      {
        src: '/projects/images/alphazero-1.5.png',
        title: 'AlphaZero Chess Concepts (third-party)',
        description:
          'Third-party figure on chess concepts learned by AlphaZero — not from this project.',
      },
    ],
  },




























  

  {
    title: 'Multilingual POS Tagging & Context-Aware Spell Correction',
    slug: 'nlp-pos-spellcheck',
    sourceCode: 'https://github.com/Vishal-Krishna-Kumar/nlp-sequence-labeling-autocorrection',
    liveUrl: 'https://pos-engine.vishalkrishnakkr.workers.dev/blog/nlp-sequence-labeling-autocorrection/',
    year: 2024,

   description: `Part-of-speech tagging and spelling correction across 3 languages (English, Japanese, Bulgarian).
        Compared a hidden Markov model tagger (Viterbi decoding) with RNN, LSTM, and BiLSTM taggers.
        A bigram HMM reached a 5.4% English word error rate.
         <br/><br/>

        Key Facts:<br/>
        <ul>
            <li>🧠 Models: HMM with Viterbi decoding; RNN, LSTM, BiLSTM.</li>
            <li>🌍 Languages: English, Japanese, Bulgarian.</li>
            <li>📊 Result: 5.4% English word error rate for a bigram HMM.</li>
            <li>✍️ Spelling correction.</li>
        </ul>


`,
role: `
Project — POS tagging and spelling correction <br/>
<ul>
  <li>🧠 Implemented and compared HMM/Viterbi, RNN, LSTM, and BiLSTM taggers.</li>
  <li>📊 A bigram HMM reached a 5.4% English word error rate; evaluated on 3 languages.</li>
</ul>
`
    ,

    techStack: ['Python', 'HMM / Viterbi', 'RNN', 'LSTM', 'BiLSTM'],

    thumbnail: '/projects/thumbnail/nlp-1.png',
    longThumbnail: '/projects/long/nlp-1.png',

    images: [
      // {
      //   src: '/projects/images/nlp-1.1.png',
      //   title: 'POS Tagging Overview',
      //   description:
      //     'Sequence labeling pipeline with HMM and neural baselines for multilingual tagging.',
      // },
      {
        src: '/projects/images/nlp-1.2.png',
        title: 'POS Tagging Example',
        description:
          'Illustration of part-of-speech tags on an example sentence.',
      },
      {
        src: '/projects/images/nlp-1.3.png',
        title: 'spaCy Example (illustrative)',
        description:
          'Illustrative spaCy dependency-parse snippet — not code from this project.',
      },
      {
        src: '/projects/images/nlp-1.4.png',
        title: 'Learning Curves',
        description:
          'Error rate by word and by sentence vs training-set size for the HMM tagger.',
      },
    ],
  },


























  {
    title: 'Interactive 3D Game Website',
    slug: 'animation-website',
    sourceCode: 'https://github.com/Vishal-Krishna-Kumar/3D-GamingSite',
    liveUrl: 'https://3d-gamingsite.netlify.app/',
    year: 2025,

    description: `Immersive 3D web experience with real-time rendering, physics simulations, and interactive animations. 
        Built with modern web technologies achieving 90 FPS on mid-range devices and <1.5s load time.
         <br/><br/>

        Key Features:<br/>
        <ul>
            <li>🎮 Scroll-Based Animations: Built dynamic, scroll-triggered motion using GSAP for a highly engaging browsing experience.</li>
            <li>🧩 Clip-Path Transitions: Implemented geometric shape reveals and section transitions using CSS clip-path animations.</li>
            <li>🕹️ 3D Hover Effects: Added interactive 3D transforms that respond to cursor/hover for a modern “gaming” feel.</li>
            <li>🎥 Video Storytelling: Integrated seamless video transitions to enhance narrative flow and visual immersion.</li>
        <li>📱 Fully Responsive UI: Optimized layout and interactions for mobile, tablet, and desktop with consistent performance.</li>
        </ul><br/>

        Architecture Overview:<br/>
        <ul>
            <li>Component-Driven UI: Modular React components (reusable sections, cards, and motion wrappers).</li>
            <li>Animation Layer (GSAP): ScrollTrigger-based timelines controlling entrance effects, transitions, and parallax motion.</li>
            <li>Styling System (Tailwind): Utility-first styling with consistent spacing, typography, and responsive breakpoints.</li>
            <li>Reusable Motion Patterns: Centralized animation utilities and shared components (e.g., rounded corners, clip masks).</li>
        </ul><br/>

        Technical Highlights:<br/>
        <ul>
            <li>Developed with React.js + Tailwind CSS + GSAP (ScrollTrigger) for smooth, high-FPS interactions</li>
            <li>Implemented clean folder structure and reusable UI primitives to keep the codebase maintainable</li>
            <li>Optimized media rendering and transitions for better perceived performance and smooth playback</li>
            <li>SEO Score 95/100 Cumulative Layout Shift 0.05 and Time to interactive is 1.2 sec</li>
        </ul>

`,
role: `
Frontend Developer & UI Engineer <br/>
Personal Project – Award-Winning Website Recreation
<ul>
  <li>🎮 Independently recreated a visually rich, award-winning 3D gaming website inspired by Zentry.</li>
  <li>🧩 Designed and implemented complex scroll-based animations and geometric transitions using GSAP.</li>
  <li>🕹️ Built interactive 3D hover effects and motion-driven UI components with React and CSS transforms.</li>
  <li>🎥 Integrated video transitions and storytelling elements for an immersive user experience.</li>
  <li>📐 Architected a reusable, component-driven frontend using React and Tailwind CSS.</li>
  <li>📱 SEO Score 95/100 Cumulative Layout Shift 0.05 and Time to interactive is 1.2 sec .</li>
</ul>
`
,

    techStack: ['React', 'Three.js', 'GSAP', 'WebGL', 'Vite', 'TypeScript', 'Tailwind CSS'],

    thumbnail: '/projects/thumbnail/3dgamingsite-1.webp',
    longThumbnail: '/projects/long/3dgamingsite-1.webp',

    images: [
      {
        src: '/projects/images/3dgamingsite-1.1.webp',
        title: 'Landing Experience',
        description:
          'High-impact hero section with layered motion and smooth entrance animations.',
      },
      {
        src: '/projects/images/3dgamingsite-1.2.webp',
        title: 'Scroll-Based Storytelling',
        description:
          'ScrollTrigger-driven timelines powering section reveals, parallax, and transitions.',
      },
      {
        src: '/projects/images/3dgamingsite-1.3.gif',
        title: 'Micro-Interactions',
        description:
          'Hover + cursor-based 3D transforms for modern interactive depth and responsiveness.',
      },
      {
        src: '/projects/images/3dgamingsite-1.4.webp',
        title: 'Responsive Layout',
        description:
          'Mobile-first layout tuned for performance, readability, and consistent motion behavior.',
      },
    ],
  },




























  {
    title: 'Combat Algo: FPS AI Game System',
    slug: 'fps-ai-render-game',
    sourceCode: 'https://github.com/Vishal-Krishna-Kumar/Combat-Algo-Game',
    liveUrl: 'https://vishalkrishna.itch.io/combat',
    year: 2024,

     description: `A high-performance Combat-style first-person shooter built with Python and Pygame, featuring AI-controlled enemies using shortest-path algorithms, intelligent spawn balancing, and real-time 
        ray-casting for immersive 3D gameplay.
         <br/><br/>

        Key Features:<br/>
        <ul>
            <li>🎮 Combat-Style 2D Raycasting Engine: Built a real-time ray-casting renderer inspired by Wolfenstein 3D for fast, immersive first-person gameplay.</li>
            <li>🔫 Core FPS Mechanics: Designed responsive shooting, health, ammo systems, pickups, and combat feedback.</li>
            <li>🕹️ WASD + Mouse Controls: Implemented smooth keyboard movement and mouse-driven camera rotation for classic FPS fee.</li>
            <li>🤖 AI-Driven Enemies: Developed enemy AI that actively hunts the player using shortest-path navigation.</li>
            <li>⚖️ Dynamic Spawn Balancing: Controlled enemy spawn rates to maintain fair difficulty and stable performance(challenging to achieve it).</li>
        </ul><br/>

        Architecture Overview:<br/>
        <ul>
            <li>Raycasting Engine Core: Casts rays per frame to calculate wall intersections and depth-correct rendering.</li>
            <li>Game Loop & Systems Layer: Central loop handling input, physics, AI updates, rendering, and combat logic.</li>
            <li>Enemy AI Module: Grid-based A* pathfinding for intelligent navigation and player pursuit.</li>
            <li>Resource Pipeline: Organized textures, sprites, sounds, and maps for efficient loading and reuse.</li>
        </ul><br/>

        Technical Highlights:<br/>
        <ul>
            <li>Developed entirely in Python using Pygame for real-time rendering and input handling.</li>
            <li>Implemented A* pathfinding for enemy navigation and obstacle avoidance.</li>
            <li>Optimized ray-casting and update loops for consistent frame rates.</li>
            <li>Packaged a standalone Windows executable using PyInstaller.</li>
        </ul>

`,
role: `
Game Developer & AI Engineer <br/>
Personal Project – Combat -Style 3D Raycasting FPS
<ul>
  <li>🎮 Independently developed a high-performance Wolfenstein-inspired 3D FPS using Python and Pygame.</li>
  <li>🧠 Designed and implemented a real-time ray-casting engine to simulate immersive 3D environments.</li>
  <li>🤖 Built AI-driven enemies using shortest-path navigation (A* pathfinding) for intelligent player pursuit.</li>
  <li>🔫 Implemented core FPS mechanics including mouse-based aiming, WASD movement, shooting, health, and pickups.</li>
  <li>⚖️ Engineered dynamic spawn balancing to maintain fair difficulty and stable performance.</li>
</ul>
`

,

    techStack: ['Python', 'Pygame', 'Ray Casting', 'A* Pathfinding', 'OOP', 'Vector Math', 'AI Behavior Trees'],

    thumbnail: '/projects/thumbnail/combat-1.png',
    longThumbnail: '/projects/long/combat-1.png',

    images: [
      {
        src: '/projects/images/combat-1.1.png',
        title: 'Ray-Casting Renderer',
        description:
          'Depth-correct wall rendering using per-frame ray intersection for a classic FPS feel.',
      },
      {
        src: '/projects/images/combat-1.2.png',
        title: 'Enemy Navigation',
        description:
          'A* pathfinding on a grid map to pursue the player and avoid obstacles.',
      },
      {
        src: '/projects/images/combat-1.3.png',
        title: 'Gameplay Systems',
        description:
          'Health/ammo, pickups, and combat feedback integrated into a stable real-time loop.',
      },
      {
        src: '/projects/images/combat-1.4.png',
        title: 'Level & Asset Pipeline',
        description:
          'Organized textures, sprites, and map assets for efficient loading and reuse.',
      },
    ],
  },
];


// Employment only. Forage job simulations are listed separately in MY_JOB_SIMULATIONS (not employment).
export const MY_EXPERIENCE = [
  {
    title: 'AI Engineer Intern',
    company: 'Hirello.ai — Remote',
    duration: 'July 2026 – September 2026',
    description:
      'Built a resume-grounded RAG application-autofill feature for job-portal forms. Contributed code to 5 of the platform\'s 10 agents (Resume Optimizer, LinkedIn Labs, Outreach, Action Academy, Chat with Hiro). Fixed 3 Resume Optimizer defects, repaired Gmail, Outlook and Yahoo OAuth connection problems, fixed an IDOR access-control vulnerability with regression tests, and added cross-browser end-to-end tests covering 3 browsers. Worked in a 7-person product and engineering team across 10 one-week sprints. Stack: Python, FastAPI, SQLAlchemy, MySQL, JavaScript, Vue 3, Vertex AI/Gemini, Pinecone.',
    thumbnail: '/projects/thumbnail/hirello-1.png',
    longThumbnail: '/projects/long/hirello-1.png',
  },
  {
    title: 'Graduate Researcher — Applied ML and Intelligent Systems',
    company: 'City University of Seattle',
    duration: 'August 2025 – Present',
    description:
      'Developed a modular RAG pipeline using dense and sparse retrieval, structured chunking and graph reasoning, reducing LLM hallucinations by 30%. Built reproducible experiment workflows, reducing debugging and rerun time by 40%. Worked with PyTorch computer-vision and NLP models.',
    thumbnail: '/projects/thumbnail/rag-1.jpg',
    longThumbnail: '/projects/long/rag-1.jpg',
  },
  {
    title: 'Graduate Teaching Assistant (part-time) — Computer Networks',
    company: 'Illinois Institute of Technology',
    duration: 'January 2025 – May 2025',
    description:
      'Supported networking labs and graded assignments. Assisted students using the FABRIC testbed. Helped configure and troubleshoot multiple servers.',
  },
];

// Job simulations completed on Forage. These are NOT employment or internships.
export const MY_JOB_SIMULATIONS = [
  {
    title: 'Job simulation (Forage) — Data Science',
    company: 'BCG (via Forage)',
    description:
      'Customer churn prediction: random forest on 10,000+ records; recall improved from 32% to 50%.',
    thumbnail: '/projects/thumbnail/bcg-1.png',
    longThumbnail: '/projects/long/bcg-1.png',
  },
  {
    title: 'Job simulation (Forage) — Advanced Software Engineering',
    company: 'Walmart (via Forage)',
    description:
      'Implemented a heap-based priority queue in Java for shipping workflows, produced a UML class diagram for a data processor with multiple execution modes and database connectivity, and designed an ER diagram for a new accounting database based on business requirements.',
    thumbnail: '/projects/thumbnail/walmart-1.png',
    longThumbnail: '/projects/long/walmart-1.png',
  },
];
