# Knowledge Base Topic Structure

## When to Use

- Reference this structure whenever you need to understand how the book-derived knowledge base is partitioned or to plan additional content.
- Use it when assigning writing or extraction tasks so each contributor knows exactly which sources and concepts belong to their file.
- Consult it before creating new documentation to avoid duplication and ensure alignment with the established taxonomy.
- Apply it as a checklist during QA to confirm each topic file includes the promised scope (theoretical background, code, pitfalls, etc.).
- Skip it if you are already working inside a specific topic file and do not need to revisit the master organization.

## Final Organization: 10 Topic Files

### File 1: Financial Data Structures and Engineering
**Sources:**
- Lopez de Prado: Chapter 2 (Financial Data Structures)
- Jansen: Chapters 2-3 (Market/Fundamental Data, Alternative Data)

**Content:**
- Types of financial data (fundamental, market, analytics, alternative)
- Standard bars vs information-driven bars (time, tick, volume, dollar bars)
- Multi-product series handling
- Sampling techniques
- Data storage and management
- Alternative data sources and evaluation

### File 2: Labeling Methods and Meta-Labeling
**Sources:**
- Lopez de Prado: Chapter 3 (Labeling)
- Jansen: Chapter 4 (Alpha Factors)

**Content:**
- Fixed-time horizon method
- Triple-barrier method
- Dynamic thresholds
- Meta-labeling approach
- Learning side and size
- Quantamental approach
- Feature engineering for labels

### File 3: Sample Weights and Data Sampling Techniques
**Sources:**
- Lopez de Prado: Chapter 4 (Sample Weights)
- Lopez de Prado: Chapter 5 (Fractionally Differentiated Features)

**Content:**
- Overlapping outcomes
- Concurrent labels and uniqueness
- Sequential bootstrap
- Return attribution
- Time decay
- Class weights
- Fractional differentiation for stationarity

### File 4: Machine Learning Models and Ensemble Methods
**Sources:**
- Lopez de Prado: Chapter 6 (Ensemble Methods)
- Jansen: Chapters on ML models (various)

**Content:**
- Bootstrap aggregation
- Random forests
- Boosting methods
- Bagging vs boosting in finance
- Deep learning for trading
- Reinforcement learning applications
- Model selection and evaluation

### File 5: Cross-Validation and Feature Importance
**Sources:**
- Lopez de Prado: Chapters 7-8 (Cross-Validation, Feature Importance)
- Lopez de Prado: Chapter 9 (Hyper-Parameter Tuning)

**Content:**
- Purged K-fold cross-validation
- Embargo techniques
- Mean decrease impurity/accuracy
- Single feature importance
- Orthogonal features
- Parallelized vs stacked importance
- Grid search and randomized search
- Scoring methods

### File 6: Backtesting Frameworks and Methodologies
**Sources:**
- Lopez de Prado: Chapters 11-13 (Dangers of Backtesting, CV Backtesting, Synthetic Data)
- Jansen: Backtesting chapters

**Content:**
- Common backtesting pitfalls
- Walk-forward analysis
- Combinatorial purged cross-validation
- Backtesting on synthetic data
- Addressing backtest overfitting
- Strategy selection
- Implementation considerations

### File 7: Performance Metrics and Risk Management
**Sources:**
- Lopez de Prado: Chapters 14-15 (Backtest Statistics, Strategy Risk)
- Lopez de Prado: Chapter 10 (Bet Sizing)

**Content:**
- Time-weighted rate of return
- Sharpe ratio variants (probabilistic, deflated)
- Drawdown and time under water
- Implementation shortfall
- Efficiency statistics
- Classification scores
- Bet sizing from probabilities
- Probability of strategy failure

### File 8: Portfolio Construction and Asset Allocation
**Sources:**
- Lopez de Prado: Chapter 16 (ML Asset Allocation)
- Jansen: Portfolio optimization chapters

**Content:**
- Hierarchical risk parity
- Markowitz's curse
- Tree clustering and quasi-diagonalization
- Recursive bisection
- Convex optimization problems
- Monte Carlo simulations
- Position sizing strategies

### File 9: Advanced Features and Market Microstructure
**Sources:**
- Lopez de Prado: Chapters 17-19 (Structural Breaks, Entropy, Microstructure)
- Jansen: Market microstructure content

**Content:**
- Structural break detection (CUSUM, explosiveness tests)
- Shannon's entropy and applications
- Lempel-Ziv estimators
- Price sequences (tick rule, Roll model)
- Strategic trade models (Kyle's lambda, Amihud's lambda)
- Sequential trade models (PIN, VPIN)
- Order flow analysis
- High-low volatility estimators

### File 10: Implementation and High-Performance Computing
**Sources:**
- Lopez de Prado: Chapters 20-22 (Multiprocessing, Quantum Computing, HPC)
- Jansen: Implementation chapters

**Content:**
- Vectorization techniques
- Multiprocessing vs multithreading
- Atoms and molecules approach
- Asynchronous processing
- Combinatorial optimization
- HPC hardware and software
- Production deployment
- Monitoring and maintenance
- Research workflow

## Processing Approach

Each file will contain:
1. Comprehensive theoretical background
2. Mathematical formulations and algorithms
3. Implementation considerations
4. Code examples and pseudocode
5. Practical applications and use cases
6. Common pitfalls and best practices
7. References to specific book sections

