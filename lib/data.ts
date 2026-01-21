import { IProject } from '@/types';
// import ProjectVideoCarousel from '@/components/ProjectVideoCarousel';


export const GENERAL_INFO = {
    email: 'vishalkrishnakkr@gmail.com',
    linkedin : "https://www.linkedin.com/in/vishal-krishna-kumar-65583a201/",

    emailSubject: "Let's collaborate on a project",
    emailBody: 'Hi Vishal, I am reaching out to you because...',

    oldPortfolio: 'https://www.vishal-krishna.me/',
    upworkProfile: '',
};

export const SOCIAL_LINKS = [
    { name: 'Github', url: 'https://github.com/Vishal-Krishna-Kumar' },
    { name: 'Linkedin', url: 'https://www.linkedin.com/in/vishal-krishna-kumar-65583a201/' },
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
        {
            name: 'MySQL',
            icon: '/logo/mysql.svg',
        },
        {
            name :"Vector DataBase",
            icon : "/logo/vector-db.png",
        },
        {
            name: 'PostgreSQL',
            icon: '/logo/postgreSQL.png',
        },
        {
            name: 'MongoDB',
            icon: '/logo/mongodb.svg',
        },
    ],

    tools: [
        {
            name: 'Git',
            icon: '/logo/git.png',
        },
        {
            name: 'Docker',
            icon: '/logo/docker.svg',
        },
        {
            name: 'AWS',
            icon: '/logo/aws.png',
        },
        
        {
            name : "Neo4j",
            icon : "/logo/neo4j.png",
        },
        {
            name: "Vercel",
            icon: "/logo/vercel.png",
        }
    ],
};

export const PROJECTS: IProject[] = [
    {
        title: 'GraphAugmented Intelligence',
        slug: 'GAI',
        sourceCode : "https://github.com/Vishal-Krishna-Kumar/GraphAugmented-Intelligence_GAI",
        // liveUrl: 'https://electroev.co.uk/',
        year: 2025,
        
        description: `
        🏆 <strong>1st Place – SCE Hacks 2025</strong> <br/>

        Designed and implemented GraphAugmented Intelligence (GAI), a prompt-aware, knowledge graph–driven retrieval-augmented generation framework that enhances factual grounding and reduces hallucinations in LLMs by 40%. 
        The system integrates semantic graph embeddings with multi-hop reasoning paths to improve answer accuracy across complex, multi-domain queries. <br/> <br/>
      
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
        techStack: ["PyTorch", "KG-RAG", "Neo4j", "FastAPI", "React", "Docker", "Kubernetes", "GraphQL"],
        demoViedo: ['/projects/images/gaidemo-1.gif', '/projects/images/gaidemo-2.mp4', '/projects/images/gaidemo-3.mp4', '/projects/images/gaidemo-4.mp4'],

        thumbnail: '/projects/thumbnail/kg-rag-1.jpeg',
        longThumbnail: '/projects/long/kg-rag-1.jpeg',
        images: [
            '/projects/images/gai-1.1.webp',
            '/projects/images/gai-1.2.png',
            '/projects/images/gai-1.3.png',
            
        ],
    },











    {
        title: '3D Attention UNet++ Brain Tumor Segmentation',
        slug: '3d-unet-brain-tumor',
        sourceCode : "https://github.com/Vishal-Krishna-Kumar/BRaTS-attention-Tumor-Segmentation-UNet-",
        liveUrl: 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11929897/',
        year: 2025,
        // demoViedo: [],

        description: `
        🏆 <strong>2nd Place, Hack for Research 2025</strong> <br/>

        Advanced 3D Attention UNet framework for multimodal MRI brain tumor segmentation and survival prediction, designed for precise volumetric analysis of gliomas. The model leverages attention mechanisms to selectively focus on tumor-relevant regions, achieving high Dice similarity and robust boundary delineation on the BraTS dataset.<br/><br/>

        Key Features:<br/>
        <ul>
            <li>🧠 3D Attention UNet Architecture: Integrates attention gates into UNet3D to suppress irrelevant regions and enhance tumor-focused feature learning.</li>
            <li>🧬 Multi-Modal MRI Fusion: Supports FLAIR, T1, T1CE, and T2 modalities for comprehensive tumor representation.</li>
            <li>🎯 High Segmentation Accuracy: Achieved Dice scores > 0.97 across tumor sub-regions (WT, TC, ET).</li>
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
        <li>✅ Achieved Dice Score > 0.97 across all tumor sub-regions</li>
        <li>🎨 Accuracy of 20% improve combining multimodel COUNCIL</li>
        <li>🔄 Sensitivity of 96.5% and neglecting the noise in segmentation</li>
        <li>🖥️ Inference Time of 0.8 seconds per volume</li>
      </ul>
      `,
    
        techStack: ["PyTorch", "3D CNNs", "BraTS Dataset", "MONAI", "NumPy", "Matplotlib", "ITK"],

        thumbnail: '/projects/thumbnail/3d-attention-1.jpg',
        longThumbnail: '/projects/long/3d-attention-1.jpg',
        images: [
            '/projects/images/3d-attention-1.1.png',
            '/projects/images/3d-attention-1.2.png',
            '/projects/images/3d-attention-1.3.png',
            '/projects/images/3d-attention-1.4.png',
        ],
       
    },











    {
        title:'DeepFake Detection System',
        slug: 'deepfake-detection',
        sourceCode : "https://github.com/Vishal-Krishna-Kumar/DeepFake-Detection-System",
        liveUrl: 'https://deep-fake-detection-paper.tiiny.site',
        year: 2024,
        //  demoViedo: [],
        description: `Advanced deepfake detection framework using ensemble of XceptionNet, EfficientNet, and Vision Transformers. Achieved state-of-the-art 94.2% accuracy on DFDC dataset with 2.1% false positive rate. System includes real-time video analysis and explainable AI visualizations.<br/><br/>

        Key Features:<br/>
        <ul>
            <li>🕵️ Deepfake Detection Pipeline: End-to-end system for detecting manipulated videos using CNN and Transformer-based architectures.</li>
            <li>🧠 Multi-Model Architecture Support: Includes ResNet, Xception, EfficientNet, MesoNet, GramNet, ViT, and M2TR.</li>
            <li>🎯 High Detection Accuracy: Achieved up to 99.9% accuracy on FF-DF and 99.7% on Celeb-DF datasets.</li>
            <li>🧬 Frequency & Attention-Based Learning: Leverages spatial, temporal, and frequency-domain cues for robust forgery detection.</li>
            <li>⚙️ Research & Production Ready: Modular design for training, evaluation, benchmarking, and deployment.</li>
        </ul>
        <br/>

        Architecture Overview:<br/>
        <ul>
            <li>Video Processing Pipeline: Real/Fake Video → Frame Extraction → Preprocessing → Model Inference.</li>
            <li>CNN-Based Models: Capture spatial artifacts and texture inconsistencies in manipulated frames.</li>
            <li>Transformer-Based Models: Learn long-range dependencies and global manipulation patterns.</li>
            <li>Unified Evaluation Framework: Standardized benchmarking across FF-DF (RAW, C23, C40) and Celeb-DF datasets.</li>
        </ul>
        <br/>

        Technical Highlights:<br/>
        <ul>
            <li>Implemented in PyTorch with configurable YAML-based experiment setup</li>
            <li>Advanced preprocessing including face alignment, normalization, and frame sampling</li>
            <li>Supports training, evaluation, visualization, and single-image/video inference</li>
            <li>Integrated performance metrics: Accuracy, AUC, Precision, Recall, and F1-score</li>
</ul>

`,
role: `
      Team Junior Lead <br/>:
      Collaborate with Wisen Platform :
      <ul>
        <li>🧠 Led development and evaluation of a 3D Attention UNet model for multimodal MRI tumor segmentation.</li>
        <li>📈 Improved overall segmentation accuracy by ~20% through multi-model ensemble (Council) strategies.</li>
        <li>🤝 Coordinated model experiments, result validation, and research alignment within the team.</li>

      </ul>
      `,
    

        techStack: ["TensorFlow", "XceptionNet", "OpenCV", "FFmpeg", "Flask", "React", "Docker"],

        thumbnail: '/projects/thumbnail/deepfake-1.png',
        longThumbnail: '/projects/long/deepfake-1.png',
        images: [
            '/projects/images/deepfake-1.1.png',
            '/projects/images/deepfake-1.2.png',
            '/projects/images/deepfake-1.3.gif',
        ],
    },













    {
        title: 'AlphaZero-Inspired Reinforcement Learning System',
        slug: 'alpha-zero-rl',
        sourceCode : "https://github.com/Vishal-Krishna-Kumar/AlphaZero-Chess",
        liveUrl: "https://alpha-zero-paper.tiiny.site/",
        year: 2024,
        //  demoViedo: [],

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





    techStack: ["PyTorch", "TensorFlow", "MCTS", "OpenAI Gym", "Ray", "Docker", "CUDA"],

        thumbnail: '/projects/thumbnail/alphazero-1.jpg',
        longThumbnail: '/projects/long/alphazero-1.jpg',
        images: [
            '/projects/images/alphazero-1.1.jpg',
            '/projects/images/alphazero-1.2.png',
            '/projects/images/alphazero-1.3.jpg',
            '/projects/images/alphazero-1.4.png',
            '/projects/images/alphazero-1.5.png',
        ],
    },






















    {
        title: 'Multilingual POS Tagging & Context-Aware Spell Correction-NLP',
        slug: 'nlp-pos-spellcheck',
        sourceCode : "https://github.com/Vishal-Krishna-Kumar/nlp-sequence-labeling-autocorrection",
        liveUrl : "https://pos-engine.vishalkrishnakkr.workers.dev/blog/nlp-sequence-labeling-autocorrection/",

        year: 2024,
        //  demoViedo: [],
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
    techStack: ["spaCy", "Transformers", "NLTK", "Scikit-learn", "PyTorch", "FastAPI"],
        
    thumbnail: '/projects/thumbnail/nlp-1.png',
        longThumbnail: '/projects/long/nlp-1.png',
        images: [
            '/projects/images/nlp-1.1.png',
            '/projects/images/nlp-1.2.png',
            '/projects/images/nlp-1.3.png',
            '/projects/images/nlp-1.4.png',
        ],

    },




















    {
        title: 'Interactive 3D Game Website',
        slug: 'animation-website',
        sourceCode:"https://github.com/Vishal-Krishna-Kumar/3D-GamingSite",
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
        
        techStack: ["React", "Three.js", "GSAP", "WebGL", "Vite", "TypeScript", "Tailwind CSS"],

        thumbnail: '/projects/thumbnail/3dgamingsite-1.webp',
        longThumbnail: '/projects/long/3dgamingsite-1.webp',
        images: [
            '/projects/images/3dgamingsite-1.1.webp',
            '/projects/images/3dgamingsite-1.2.webp',
            '/projects/images/3dgamingsite-1.3.gif',
            '/projects/images/3dgamingsite-1.4.webp',
        ]
    },
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     
    
    
    {
        title: 'Combat Algo AI Game System',
        slug: 'fps-ai-render-game',
        sourceCode:"https://github.com/Vishal-Krishna-Kumar/Combat-Algo-Game",
        liveUrl: 'https://vishalkrishna.itch.io/combat',
        year: 2024,
        // demoViedo: [],
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
        
        techStack: [
    "Python",
    "Pygame",
    "Ray Casting",
    "A* Pathfinding",
    "OOP",
    "Vector Math",
    "AI Behavior Trees"
  ],

        thumbnail: '/projects/thumbnail/combat-1.png',
        longThumbnail: '/projects/long/combat-1.png',
        images: [
            '/projects/images/combat-1.1.png',
            '/projects/images/combat-1.2.png',
            '/projects/images/combat-1.3.png',
            '/projects/images/combat-1.4.png',
        ]
    },

];

export const MY_EXPERIENCE = [
    {
        title: "Advanced Software Engineering Virtual Experience . ",
        company: 'Walmart',
        description : "Completed advanced software engineering simulations across multiple Walmart teams. Implemented a novel heap data structure in Java for shipping workflows, demonstrating strong algorithmic problem-solving. Produced a UML class diagram for a data processor supporting multiple operating modes and database connections, and designed an ER diagram for a new accounting database based on business requirements.",
        thumbnail: '/projects/thumbnail/walmart-1.png',
        longThumbnail: '/projects/long/walmart-1.png',
    },
    {
        title: 'BCG - Data Science intern Virtual . ',
        company: 'BCG',
        description : "Engineered an end-to-end data pipeline for a customer churn simulation, involving rigorous data cleaning, feature engineering of transactional drivers, and predictive modeling. Employed structured analytical thinking to validate model accuracy and ensure findings were rooted in statistically significant behavioral triggers.",
        thumbnail: '/projects/thumbnail/bcg-1.png',
        longThumbnail: '/projects/long/bcg-1.png',
        // duration: 'Oct 2023 - Nov 2024',
    },
    {
        title: 'Research Assistant  Intelligent Systems . ',
        company: 'Intelligent Systems Lab',
        description:
      "Leading research in Retrieval-Augmented Generation (RAG) and graph-based reasoning systems. Developed modular RAG pipelines with semantic chunking and hybrid retrieval, reducing hallucinations by ~35%. Implemented computer vision models using PyTorch, achieving ~25% performance gains through architecture optimization.Contributed to open-source AI frameworks.",
        thumbnail: '/projects/thumbnail/rag-1.jpg',
        longThumbnail: '/projects/long/rag-1.jpg',
        // duration: 'Oct 2022 - Sep 2023',
    },
];
