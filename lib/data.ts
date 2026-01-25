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
    { name: 'Vector Databases', icon: '/logo/vector-db.png' },
    { name: 'PostgreSQL', icon: '/logo/postgreSQL.png' },
    { name: 'MongoDB', icon: '/logo/mongodb.svg' },
  ],

  tools: [
    { name: 'Git', icon: '/logo/git.png' },
    { name: 'Docker', icon: '/logo/docker.svg' },
    { name: 'AWS', icon: '/logo/aws.png' },
    { name: 'Neo4j', icon: '/logo/neo4j.png' },
    { name: 'Vercel', icon: '/logo/vercel.png' },
  ],
};

export const PROJECTS: IProject[] = [
  {
    title: 'GraphAugmented Intelligence',
    slug: 'GAI',
    sourceCode: 'https://github.com/Vishal-Krishna-Kumar/GraphAugmented-Intelligence_GAI',
    year: 2025,

     description: `
        🏆 <strong>1st Place – SCE Hacks 2025</strong> <br/>

        Designed and implemented Graph-Augmented Intelligence (GAI), a prompt-aware, knowledge graph–driven Retrieval-Augmented Generation (RAG) framework that improves factual grounding and significantly reduces hallucinations in Large Language Models (LLMs). The system combines structured knowledge graphs with LLM reasoning to enable accurate, low-latency responses for complex, multi-hop queries.<br/><br/>

        GAI dynamically extracts minimal, prompt-aware contextual subgraphs from a knowledge graph and injects them into the LLM generation pipeline, enabling grounded reasoning across multi-domain information. The framework was validated on large-scale biomedical knowledge graphs, supporting reasoning over entities such as diseases, drugs, genes, and biological relationships. <br/><br/>
      
      Key Features:<br/>
      <ul>
        <li>🧠Hallucination Reduction: 40% reduction in hallucinated responses.</li>
        <li>🔗Multi-hop Reasoning: Enhanced accuracy in complex queries achieving 35% improvement.</li>
        <li>🛒 Response Latency: < 2s average response time.</li>
        <li>📱 Knowledge Recall: 92% accuracy in retrieving relevant information.</li>
      </ul><br/>

      Architecture Overview:<br/>
        <ul>
        <li>Knowledge Graph Construction: Integrated multi-domain data into a unified knowledge graph using Neo4j.</li>
        <li>Semantic Embeddings: Employed graph neural networks to generate context-aware embeddings for nodes and relationships.</li>
        <li>Prompt-Aware Retrieval: Developed a retrieval mechanism that leverages prompt context to fetch relevant graph segments.</li>
        <li>LLM Integration: Combined retrieved graph data with LLMs (e.g., GPT-4) for response generation.</li>
        <li>Evaluation Framework: Established metrics for assessing hallucination rates, response accuracy, and latency.</li>

        </ul><br/>
      
      Technical Highlights:
      <ul>
        <li>Implemented a robust knowledge graph architecture using Neo4j and graph neural networks.</li>
        <li>Multi-model query support for diverse data sources.</li>
        <li>Explainable AI with reasoning capabilities.</li>
        <li>Automated graph maintenance and updates Knoweledge graph.</li>
      </ul>
      `,
        role: `
      Team Lead <br/>
      Owned the entire development lifecycle:
      <ul>
        <li>✅ Real time knowledge graph updates and synchronization.</li>
        <li>🎨 Multi Modal Data integration.</li>
        <li>🔄 Scalability to million of nodes and relationships.</li>
        <li>🖥️ Federated Graph Updates and learning.</li>
      </ul>
      `,

    techStack: [ 'PyTorch', 'KG-RAG','Neo4j','FastAPI','React','Docker','Kubernetes','GraphQL',],


    demoViedo: [
      '/projects/images/gaidemo-1.gif',
      '/projects/images/gaidemo-2.mp4',
      '/projects/images/gaidemo-3.mp4',
      '/projects/images/gaidemo-4.mp4',
    ],

    thumbnail: '/projects/thumbnail/kg-rag-1.webp',
    longThumbnail: '/projects/long/kg-rag-1.webp',

    images: [
      {
        src: '/projects/images/gai-1.1.webp',
        title: 'GAI System Overview',
        description:
          'End-to-end flow from prompt → subgraph retrieval → evidence injection → grounded LLM response.',
      },
      {
        src: '/projects/images/gai-1.2.png',
        title: 'Knowledge Graph Construction',
        description:
          'Multi-source ingestion and schema design for biomedical entities and relationships using Neo4j.',
      },
      {
        src: '/projects/images/gai-1.3.png',
        title: 'Prompt-Aware Retrieval',
        description:
          'Minimal, context-relevant subgraph extraction to support multi-hop reasoning with explainable evidence.',
      },
    ],
  },





























  {
    title: '3D Attention UNet++ Brain Tumor Segmentation',
    slug: '3d-unet-brain-tumor',
    sourceCode: 'https://github.com/Vishal-Krishna-Kumar/BRaTS-attention-Tumor-Segmentation-UNet-',
    liveUrl: 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11929897/',
    year: 2025,

     description: `
        🏆 <strong>2nd Place, Hack for Research 2025</strong> <br/>

        Advanced 3D Attention UNet framework for multimodal MRI brain tumor segmentation and survival prediction, designed for precise volumetric analysis of gliomas. The model leverages attention mechanisms to selectively focus on tumor-relevant regions, achieving high Dice similarity and robust boundary delineation on the BraTS dataset.<br/><br/>

        Key Features:<br/>
        <ul>
            <li>🧠 3D Attention UNet Architecture: Integrates attention gates into UNet3D to suppress irrelevant regions and enhance tumor-focused feature learning.</li>
            <li>🧬 Multi-Modal MRI Fusion: Supports FLAIR, T1, T1CE, and T2 modalities for comprehensive tumor representation.</li>
            <li>🎯 High Segmentation Accuracy: Achieved Dice scores > 0.90 across tumor sub-regions (WT, TC, ET).</li>
            <li>📊 Survival Prediction Module: Includes optional regression/classification head for patient survival estimation.</li>
            <li>⚙️ Deep Supervision & Residual Learning: Improves gradient flow and stabilizes training for 3D volumes.</li>
        </ul><br/>

        Architecture Overview:<br/>
        <ul>
            <li>Encoder–Decoder UNet3D Backbone: Captures hierarchical spatial features from volumetric MRI inputs.</li>
            <li>Attention Gates: Dynamically weight skip-connection features based on tumor relevance.</li>
            <li>Multi-Scale Feature Aggregation: Preserves fine-grained tumor boundaries across resolutions.</li>
            <li>Joint Learning Setup: Enables simultaneous tumor segmentation and survival prediction.</li>
        </ul><br/>

        Technical Highlights:<br/>
        <ul>
            <li>Built with PyTorch using a modular and extensible training pipeline</li>
            <li>Advanced preprocessing: normalization, patch extraction, rotation, flipping, and intensity augmentation</li>
            <li>Evaluated using Dice Score, Hausdorff Distance, Sensitivity, and Specificity</li>
        <li>Adapted from PyTorch-3DUNet and optimized for BraTS MRI datasets</li>
        </ul>
`,

         role: `
      Team Lead <br/>
      Research and Development of the Model:
      <ul>
        <li>✅ Achieved Dice Score > 0.90 across all tumor sub-regions</li>
        <li>🎨 Accuracy of 20% improve combining multimodel COUNCIL</li>
        <li>🔄 Sensitivity of 96.5% and neglecting the noise in segmentation</li>
        <li>🖥️ Inference Time of 0.8 seconds per volume</li>
      </ul>
      `,


    techStack: ['PyTorch', '3D CNNs', 'BraTS Dataset', 'MONAI', 'NumPy', 'Matplotlib', 'ITK'],

    thumbnail: '/projects/thumbnail/3d-attention-1.jpg',
    longThumbnail: '/projects/long/3d-attention-1.jpg',

    images: [
      {
        src: '/projects/images/3d-attention-1.1.png',
        title: 'Model Architecture',
        description:
          '3D UNet++ backbone with attention gates and deep supervision for stable volumetric learning.',
      },
      {
        src: '/projects/images/3d-attention-1.2.png',
        title: 'Multimodal MRI Inputs',
        description:
          'Fusion of FLAIR/T1/T1CE/T2 modalities to capture complementary tumor signals.',
      },
      {
        src: '/projects/images/3d-attention-1.3.png',
        title: 'Segmentation Outputs',
        description:
          'Predicted tumor subregions (WT/TC/ET) with strong boundary delineation on BraTS samples.',
      },
      {
        src: '/projects/images/3d-attention-1.4.png',
        title: 'Metrics & Validation',
        description:
          'Quantitative evaluation using Dice and Hausdorff distance with sensitivity/specificity analysis.',
      },
    ],
  },




























  {
    title: 'Deepfake Detection System',
    slug: 'deepfake-detection',
    sourceCode: 'https://github.com/Vishal-Krishna-Kumar/DeepFake-Detection-System',
    liveUrl: 'https://deep-fake-detection-paper.tiiny.site',
    year: 2024,

     description: `
Designed and contributed to a scalable **Deepfake Detection framework** for identifying manipulated media using **CNN- and Transformer-based architectures**. The system provides an extensible pipeline for **training, evaluation, benchmarking, and inference** across widely used deepfake datasets, supporting both research and real-world security applications.<br/><br/>

The framework enables standardized benchmarking of multiple state-of-the-art models on **FaceForensics++ (RAW, C23, C40)** and **Celeb-DF**, allowing fair comparison across compression levels and manipulation types.<br/><br/>

Key Features:<br/>
<ul>
  <li>🕵️ <strong>End-to-End Detection Pipeline:</strong> Video → frame extraction → preprocessing → model inference → classification.</li>
  <li>🧠 <strong>Multi-Model Architecture Support:</strong> ResNet, Xception, EfficientNet, MesoNet, GramNet, F3Net, ViT, and M2TR.</li>
  <li>🎯 <strong>High Detection Performance:</strong> Achieved up to <strong>95.9% accuracy on FF-DF</strong> and <strong>94.7% on Celeb-DF</strong> across baseline models.</li>
  <li>🧬 <strong>Frequency & Attention-Based Learning:</strong> Captures spatial, temporal, and frequency-domain forgery cues.</li>
  <li>⚙️ <strong>Research-Ready Design:</strong> Modular, configurable framework for training, evaluation, visualization, and inference.</li>
</ul>
<br/>

Architecture Overview: <br/>
<ul>
  <li><strong>Video Processing Pipeline:</strong> Real/Fake video ingestion with frame extraction and preprocessing.</li>
  <li><strong>CNN-Based Models:</strong> Learn local spatial artifacts and texture inconsistencies in manipulated frames.</li>
  <li><strong>Transformer-Based Models:</strong> Capture global context and long-range manipulation patterns.</li>
  <li><strong>Unified Evaluation Framework:</strong> Standardized benchmarking across FF-DF (RAW, C23, C40) and Celeb-DF datasets.</li>
</ul>
<br/>

Technical Highlights:<br/>
<ul>
  <li>Implemented in <strong>PyTorch</strong> with YAML-based experiment configuration</li>
  <li>Advanced preprocessing including face alignment, normalization, and frame sampling</li>
  <li>Supports training, evaluation, visualization, and single-image/video inference</li>
  <li>Integrated performance metrics: Accuracy, AUC, Precision, Recall, and F1-score</li>
</ul>
`
,
role: `
      Team Junior Lead <br/>:
      Collaborate with Wisen Platform :
      <ul>
        <li>🧠 Led development and evaluation of a 3D Attention UNet model for multimodal MRI tumor segmentation.</li>
        <li>📈 Improved overall segmentation accuracy by ~20% through multi-model ensemble (Council) strategies.</li>
        <li>🤝 Coordinated model experiments, result validation, and research alignment within the team.</li>

      </ul>
      `,
    techStack: ['TensorFlow', 'XceptionNet', 'OpenCV', 'FFmpeg', 'Flask', 'React', 'Docker'],

    thumbnail: '/projects/thumbnail/deepfake-1.png',
    longThumbnail: '/projects/long/deepfake-1.png',

    images: [
      {
        src: '/projects/images/deepfake-1.1.png',
        title: 'Detection Pipeline',
        description:
          'High-level flow from video ingestion to classification with modular preprocessing and inference.',
      },
      {
        src: '/projects/images/deepfake-1.2.png',
        title: 'Benchmarking & Metrics',
        description:
          'Cross-dataset evaluation with standardized metrics and compression-level comparisons.',
      },
      {
        src: '/projects/images/deepfake-1.3.gif',
        title: 'Inference Visualization',
        description:
          'Example inference output showing frame-level processing and final prediction aggregation.',
      },
    ],
  },

































  

  {
    title: 'AlphaZero-Inspired Reinforcement Learning System',
    slug: 'alpha-zero-rl',
    sourceCode: 'https://github.com/Vishal-Krishna-Kumar/AlphaZero-Chess',
    liveUrl: 'https://alpha-zero-paper.tiiny.site/',
    year: 2024,

     description: `AlphaZero-inspired self-play reinforcement learning system with advanced Monte Carlo Tree Search (MCTS) and 
        neural policy-value networks. Achieved a ~81 % win rate against strong baseline engines and rule-based opponents, with 30% faster convergence through optimized exploration strategies.
<br/><br/>

        Key Features:<br/>
        <ul>
            <li>♟️ AlphaZero-Style Self-Play RL: Trains an agent from scratch with zero human gameplay data using iterative self-play.</li>
            <li>🌲 Neural-Guided MCTS: Combines Monte Carlo Tree Search with policy priors and value estimates for strong decision-making.</li>
            <li>🧠 Policy–Value Network: Single PyTorch model predicts (move probabilities, win probability) from game states.</li>
            <li>🔁 Replay Buffer Training Loop: Stores recent self-play games and samples mini-batches for stable learning.</li>
            <li>🧩 Multi-Game Support: Unified game API enabling Tic-Tac-Toe and Connect Four with the same training pipeline.</li>
        </ul><br/>

        Architecture Overview:<br/>
        <ul>
            <li>Game Environment Layer: Implements state, legal_moves(), apply_move(), is_terminal(), winner() for deterministic board games.</li>
            <li>MCTS Search Procedure: Uses PUCT-style selection with neural priors; outputs visit-count policy targets for learning.</li>
            <li>Self-Play Data Generation: Produces (state, MCTS-policy, outcome) triplets to train the network end-to-end.</li>
            <li>Policy + Value Optimization: Joint loss = policy cross-entropy + value MSE (+ optional regularization).</li>
        </ul><br/>

        Technical Highlights:<br/>
        <ul>
            <li>Implemented in PyTorch with checkpointing (model_*.pt) for continuous training and evaluation</li>
            <li>Data augmentation via board symmetries to improve sample efficiency and generalization</li>
            <li>Model evaluation pipeline: new checkpoints compete vs previous best/baseline to decide promotion</li>
            <li>Experiment tracking for hyperparameters, optimizers (optimizer_*.pth), and architecture variants</li>

        </ul>

`,
role: `
Independent Reinforcement Learning Engineer <br/>
Personal Research Project – Self-Play & Game AI worked with amazon Android Developer and Urjanet SDE Developer
<ul>
  <li>♟️ Designed and implemented an AlphaZero-inspired self-play reinforcement learning system from scratch.</li>
  <li>🌲 Built a neural-guided Monte Carlo Tree Search (PUCT) integrated with a policy–value network.</li>
  <li>🧠 Developed the full training loop including self-play generation, replay buffer management, and model updates.</li>
  <li>🔁 Led model evaluation by benchmarking new checkpoints against previous best agents for promotion.</li>
  <li>🧩 Extended the framework to support multiple games (Tic-Tac-Toe, Connect Four) via a unified environment API.</li>
  <li>⚙️ Conducted hyperparameter tuning and architectural experiments to improve convergence and sample efficiency.</li>
</ul>
      `,

    techStack: ['PyTorch', 'MCTS', 'OpenAI Gym', 'Ray', 'Docker', 'CUDA'],

    thumbnail: '/projects/thumbnail/alphazero-1.jpg',
    longThumbnail: '/projects/long/alphazero-1.jpg',

    images: [
      {
        src: '/projects/images/alphazero-1.1.jpg',
        title: 'Self-Play Training Loop',
        description:
          'Data generation via self-play feeding a replay buffer to train the policy–value network.',
      },
      {
        src: '/projects/images/alphazero-1.2.png',
        title: 'MCTS Search Visualization',
        description:
          'Tree expansion and visit counts guided by neural priors and value estimates (PUCT).',
      },
      {
        src: '/projects/images/alphazero-1.3.jpg',
        title: 'Policy–Value Network',
        description:
          'Single network predicting move probabilities and win likelihood from board states.',
      },
      {
        src: '/projects/images/alphazero-1.4.png',
        title: 'Evaluation & Promotion',
        description:
          'Checkpoint tournament against prior best/baselines to promote stronger models.',
      },
      {
        src: '/projects/images/alphazero-1.5.png',
        title: 'Multi-Game Support',
        description:
          'Unified game API enabling the same training pipeline to support different board games.',
      },
    ],
  },




























  

  {
    title: 'Multilingual POS Tagging & Context-Aware Spell Correction',
    slug: 'nlp-pos-spellcheck',
    sourceCode: 'https://github.com/Vishal-Krishna-Kumar/nlp-sequence-labeling-autocorrection',
    liveUrl: 'https://pos-engine.vishalkrishnakkr.workers.dev/blog/nlp-sequence-labeling-autocorrection/',
    year: 2024,

   description: `State-of-the-art multilingual POS tagging and context-aware spell correction system supporting 5+ languages. Achieved 96.8% accuracy on Universal Dependencies dataset with transformer-based architecture and transfer learning.
         <br/><br/>

        Key Features:<br/>
        <ul>
            <li>🧠 Multi-Model POS Tagging: Implemented Hidden Markov Models (Bigram, Trigram) and neural models (RNN, LSTM, BiLSTM).</li>
            <li>🌍 Multilingual NLP Support: Evaluated POS tagging across English, Japanese, and Bulgarian datasets.</li>
            <li>📊 Statistical & Neural Comparison: Benchmarked probabilistic HMMs against deep learning approaches.</li>
            <li>✍️ Autocorrection System: Built spell correction using unigram, bigram, trigram language models with smoothing and backoff.</li>
            <li>⚙️ End-to-End NLP Pipeline: Covers training, inference, evaluation, and error analysis.</li>
        </ul><br/>

        Architecture Overview:<br/>
        <ul>
            <li>HMM POS Tagger: Learned emission and transition probabilities with Viterbi decoding.</li>
            <li>Neural POS Models: Implemented Vanilla RNN, LSTM, and Bidirectional LSTM for sequence labeling.</li>
            <li>Language Modeling for Autocorrection: Utilized n-gram models combined with edit-distance-based error modeling.</li>
            <li>Evaluation Framework: Measured Error Rate by Word (ERW) and Error Rate by Sentence (ERS).</li>
        </ul><br/>

        Technical Highlights:<br/>
        <ul>
            <li>Implemented in Python with modular scripts for training, inference, and evaluation</li>
            <li>Analyzed learning curves and performance trade-offs across statistical and neural models</li>
            <li>Processed datasets(we had few amount of dataset but we did Augmentation to increase size) ranging from 13K–15K tokens across multiple languages</li>
            <li>Compared accuracy, runtime efficiency, and generalization of classical vs deep NLP models</li>
        </ul>


`,
role: `
Worked under NLP Scientist <br/>
Academic / Research-Oriented Project – POS Tagging & Autocorrection
<ul>
  <li>🧠 Designed and implemented statistical (HMM) and neural (RNN, LSTM, BiLSTM) models for part-of-speech tagging.</li>
  <li>📊 Conducted multilingual evaluation across English, Japanese, and Bulgarian datasets using ERW and ERS metrics.</li>
  <li>✍️ Built an autocorrection system using n-gram language models with smoothing, backoff, and edit-distance error modeling.</li>
  <li>🔬 Performed comparative analysis of classical NLP methods versus deep learning approaches.</li>
  <li>⚙️ Developed end-to-end NLP pipelines covering training, inference, evaluation, and error analysis.</li>
</ul>
`
    ,

    techStack: ['spaCy', 'Transformers', 'NLTK', 'Scikit-learn', 'PyTorch', 'FastAPI'],

    thumbnail: '/projects/thumbnail/nlp-1.png',
    longThumbnail: '/projects/long/nlp-1.png',

    images: [
      {
        src: '/projects/images/nlp-1.1.png',
        title: 'POS Tagging Overview',
        description:
          'Sequence labeling pipeline with HMM and neural baselines for multilingual tagging.',
      },
      {
        src: '/projects/images/nlp-1.2.png',
        title: 'Model Comparisons',
        description:
          'Benchmark results comparing classical probabilistic models vs deep learning approaches.',
      },
      {
        src: '/projects/images/nlp-1.3.png',
        title: 'Spell Correction Pipeline',
        description:
          'Context-aware correction using smoothed n-grams and edit-distance candidate generation.',
      },
      {
        src: '/projects/images/nlp-1.4.png',
        title: 'Error Analysis',
        description:
          'Word- and sentence-level error breakdown with examples and failure-mode categorization.',
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


export const MY_EXPERIENCE = [
  {
    title: 'Advanced Software Engineering Virtual Experience',
    company: 'Walmart (Forage)',
    description:
      'Completed advanced software engineering simulations across Walmart teams. Implemented a heap-based priority queue in Java for shipping workflows, produced a UML class diagram for a data processor with multiple execution modes and database connectivity, and designed an ER diagram for a new accounting database based on business requirements.',
    thumbnail: '/projects/thumbnail/walmart-1.png',
    longThumbnail: '/projects/long/walmart-1.png',
  },
  {
    title: 'Data Science Virtual Experience',
    company: 'BCG (Forage)',
    description:
      'Built an end-to-end churn analysis workflow: data cleaning, feature engineering of behavioral/transactional drivers, and predictive modeling. Applied structured evaluation and error analysis to ensure insights were supported by measurable model performance.',
    thumbnail: '/projects/thumbnail/bcg-1.png',
    longThumbnail: '/projects/long/bcg-1.png',
  },
  {
    title: 'Research Assistant (Applied ML & Intelligent Systems)',
    company: 'Intelligent Systems Lab',
    description:
      'Conducting research in Retrieval-Augmented Generation (RAG) and graph-based reasoning. Developed modular RAG pipelines with semantic chunking and hybrid retrieval, reducing hallucinations and improving grounding. Built and optimized computer vision models in PyTorch and contributed to reproducible research workflows and documentation.',
    thumbnail: '/projects/thumbnail/rag-1.jpg',
    longThumbnail: '/projects/long/rag-1.jpg',
  },
];
