"""
Labeled evaluation set for the FluxRouter complexity classifier.

100 examples (~33 per tier), NONE overlapping with the ANCHORS in complexity.py.
Domains: coding, general QA, summarisation, architecture/design, debugging,
         math, creative writing.

Format:
    EVAL_SET: list[tuple[str, Complexity]]
"""
from router.classifier.complexity import Complexity

EVAL_SET: list[tuple[str, Complexity]] = [
    # -----------------------------------------------------------------------
    # SIMPLE  (34 examples)
    # -----------------------------------------------------------------------

    # General QA — simple
    ("What is the capital of Japan?", Complexity.SIMPLE),
    ("How many continents are there?", Complexity.SIMPLE),
    ("What is the chemical symbol for gold?", Complexity.SIMPLE),
    ("What year did World War II end?", Complexity.SIMPLE),
    ("What is the tallest mountain in the world?", Complexity.SIMPLE),
    ("What does 'CPU' stand for?", Complexity.SIMPLE),
    ("Who painted the Mona Lisa?", Complexity.SIMPLE),
    ("What is the currency of Japan?", Complexity.SIMPLE),

    # Coding — simple
    ("What is the difference between '==' and '===' in JavaScript?", Complexity.SIMPLE),
    ("How do I comment out a line in Python?", Complexity.SIMPLE),
    ("What does 'return' do in a function?", Complexity.SIMPLE),
    ("Write a one-liner to reverse a string in Python.", Complexity.SIMPLE),
    ("What is the purpose of a .gitignore file?", Complexity.SIMPLE),
    ("How do I install a package with pip?", Complexity.SIMPLE),
    ("What does 'stdout' mean?", Complexity.SIMPLE),

    # Math — simple
    ("What is 7 times 8?", Complexity.SIMPLE),
    ("What is the area of a circle with radius 5?", Complexity.SIMPLE),
    ("Convert 0.75 to a percentage.", Complexity.SIMPLE),
    ("What is the prime factorization of 12?", Complexity.SIMPLE),

    # Summarisation — simple
    ("Summarize this sentence in fewer words: 'The quick brown fox jumped over the lazy dog.'", Complexity.SIMPLE),
    ("Give me a one-word label for this topic: climate, emissions, weather, temperature.", Complexity.SIMPLE),

    # Creative writing — simple
    ("Write a three-word motivational slogan.", Complexity.SIMPLE),
    ("Give me a fun fact about penguins.", Complexity.SIMPLE),
    ("Write a short greeting card message for a retirement.", Complexity.SIMPLE),
    ("Suggest a pet name for a golden retriever.", Complexity.SIMPLE),
    ("Write a two-sentence introduction for a podcast about cooking.", Complexity.SIMPLE),

    # Miscellaneous simple
    ("What is the opposite of 'transparent'?", Complexity.SIMPLE),
    ("How many ounces are in a pound?", Complexity.SIMPLE),
    ("What country has the maple leaf on its flag?", Complexity.SIMPLE),
    ("Translate 'thank you' to French.", Complexity.SIMPLE),
    ("What does the 'ls' command do in Linux?", Complexity.SIMPLE),
    ("What is a variable in programming?", Complexity.SIMPLE),
    ("Convert the number 255 to binary.", Complexity.SIMPLE),
    ("What is the first element in the periodic table?", Complexity.SIMPLE),

    # -----------------------------------------------------------------------
    # MEDIUM  (33 examples)
    # -----------------------------------------------------------------------

    # Coding — medium
    ("Write a Python decorator that measures the execution time of a function.", Complexity.MEDIUM),
    ("How do I implement pagination in a Django REST Framework API?", Complexity.MEDIUM),
    ("Write a JavaScript function that debounces user input and calls an API.", Complexity.MEDIUM),
    ("Explain the difference between a process and a thread.", Complexity.MEDIUM),
    ("How do I handle circular imports in Python?", Complexity.MEDIUM),
    ("Write a Bash script that backs up a directory with a timestamped filename.", Complexity.MEDIUM),
    ("What is memoization and how do I implement it in Python?", Complexity.MEDIUM),
    ("Write a middleware in Express.js that logs all incoming requests.", Complexity.MEDIUM),
    ("Explain the difference between eager and lazy evaluation.", Complexity.MEDIUM),
    ("How would you implement rate limiting on a single-server FastAPI application?", Complexity.MEDIUM),

    # Architecture / design — medium
    ("What are the pros and cons of microservices vs a monolith for a startup?", Complexity.MEDIUM),
    ("How should I structure environment variables for a 12-factor app?", Complexity.MEDIUM),
    ("Explain the publisher-subscriber pattern and when to use it.", Complexity.MEDIUM),
    ("What is idempotency and why is it important in API design?", Complexity.MEDIUM),
    ("How do feature flags help with continuous delivery?", Complexity.MEDIUM),

    # Debugging — medium
    ("My Python script exits with 'RecursionError: maximum recursion depth exceeded' — how do I fix it?", Complexity.MEDIUM),
    ("Why is my CSS flexbox layout not centering items vertically?", Complexity.MEDIUM),
    ("My Node.js app runs out of memory after a few hours — what should I check?", Complexity.MEDIUM),
    ("Why does my Git merge keep producing conflicts on the same file?", Complexity.MEDIUM),

    # NLP / ML — medium
    ("What is the difference between a language model and a chat model?", Complexity.MEDIUM),
    ("Explain what tokenization is and why it matters for LLMs.", Complexity.MEDIUM),
    ("What is transfer learning and why is it useful?", Complexity.MEDIUM),
    ("Explain the difference between batch norm and layer norm.", Complexity.MEDIUM),

    # Summarisation — medium
    ("Summarize the key differences between Python 2 and Python 3 in a short paragraph.", Complexity.MEDIUM),
    ("Write a concise changelog entry for a new feature that adds OAuth2 login.", Complexity.MEDIUM),
    ("Distil the main argument of this three-paragraph essay into one sentence.", Complexity.MEDIUM),

    # Math — medium
    ("What is the integral of sin(x) from 0 to π?", Complexity.MEDIUM),
    ("Explain what a p-value is in plain English and when it's misused.", Complexity.MEDIUM),
    ("How do you compute the Euclidean distance between two vectors?", Complexity.MEDIUM),

    # Creative writing — medium
    ("Write a 150-word product description for a smartwatch targeted at athletes.", Complexity.MEDIUM),
    ("Draft a polite but firm email to a vendor about a missed delivery deadline.", Complexity.MEDIUM),
    ("Write a short bio (3 sentences) for a data scientist speaking at a conference.", Complexity.MEDIUM),
    ("Create a five-item FAQ section for a SaaS pricing page.", Complexity.MEDIUM),

    # -----------------------------------------------------------------------
    # COMPLEX  (33 examples)
    # -----------------------------------------------------------------------

    # Debugging — complex
    ("Our distributed Spark job produces different results on each run despite using the same seed — diagnose.", Complexity.COMPLEX),
    ("My Go service has a goroutine leak that only appears under production traffic — how do I find and fix it?", Complexity.COMPLEX),
    ("This C++ template metaprogram compiles fine with Clang but fails with GCC — explain why and fix it.", Complexity.COMPLEX),
    ("Our Redis cluster is occasionally returning stale data after a master failover — root-cause this.", Complexity.COMPLEX),
    ("Explain why this JVM application exhibits stop-the-world GC pauses of 5+ seconds and how to eliminate them.", Complexity.COMPLEX),
    ("Our ML model shows severe performance degradation in production despite identical inputs in staging — debug.", Complexity.COMPLEX),

    # Architecture / design — complex
    ("Design a globally distributed pub/sub system that guarantees at-least-once delivery with deduplication.", Complexity.COMPLEX),
    ("Architect a real-time collaborative document editing system like Google Docs from scratch.", Complexity.COMPLEX),
    ("How would you design a search engine indexing pipeline for 100 billion documents?", Complexity.COMPLEX),
    ("Design a self-healing Kubernetes operator that detects and remediates pod failures automatically.", Complexity.COMPLEX),
    ("Architect a feature store for a large-scale ML platform that serves 50k predictions per second.", Complexity.COMPLEX),
    ("How would you implement bi-directional sync between an offline-capable mobile app and a cloud backend?", Complexity.COMPLEX),
    ("Design a chaos engineering framework that automatically injects faults and validates SLO compliance.", Complexity.COMPLEX),
    ("How would you build a multi-tenant vector database with tenant isolation and low latency?", Complexity.COMPLEX),

    # Coding — complex
    ("Implement a B+ tree with splitting, merging, and range queries.", Complexity.COMPLEX),
    ("Write a parser combinator library from scratch in Haskell.", Complexity.COMPLEX),
    ("Implement a correct and efficient topological sort that handles cyclic dependencies gracefully.", Complexity.COMPLEX),
    ("Write a concurrent work-stealing scheduler in Rust without using any async runtime.", Complexity.COMPLEX),
    ("Implement a delta CRDT for collaborative plain-text editing.", Complexity.COMPLEX),
    ("Write a bytecode interpreter for a simple stack-based virtual machine in C.", Complexity.COMPLEX),

    # Math — complex
    ("Prove that there are infinitely many primes using Euclid's proof, then explain Bertrand's postulate.", Complexity.COMPLEX),
    ("Derive the Fourier transform of a Gaussian and explain its significance in signal processing.", Complexity.COMPLEX),
    ("Explain the PAC learning framework and derive the sample complexity bound for a finite hypothesis class.", Complexity.COMPLEX),
    ("Prove the correctness of Dijkstra's algorithm using induction on the number of settled nodes.", Complexity.COMPLEX),

    # ML / AI — complex
    ("Explain how model distillation works and how to choose a good teacher-student architecture.", Complexity.COMPLEX),
    ("Design a continual learning system that avoids catastrophic forgetting for a text classification model.", Complexity.COMPLEX),
    ("How would you implement a reward model for RLHF and what biases should you watch out for?", Complexity.COMPLEX),
    ("Design a pipeline to detect and mitigate concept drift in a deployed fraud detection model.", Complexity.COMPLEX),
    ("Explain the differences between PPO, DPO, and GRPO for fine-tuning language models.", Complexity.COMPLEX),

    # Creative writing — complex
    ("Write a 500-word chapter opening for a literary novel exploring grief through unreliable narration.", Complexity.COMPLEX),
    ("Draft a comprehensive technical design document for a new open-source observability platform.", Complexity.COMPLEX),
    ("Write a philosophical essay (500 words) arguing that consciousness cannot be substrate-independent.", Complexity.COMPLEX),
    ("Create a detailed screenplay scene (one page) with subtext-heavy dialogue between two estranged siblings.", Complexity.COMPLEX),
]
