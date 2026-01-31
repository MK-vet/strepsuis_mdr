# Algorithms Documentation

This document provides detailed algorithmic descriptions and Big-O complexity analysis
for the key computational methods in the StrepSuis Suite.

## Overview

All algorithms in the StrepSuis Suite are designed with:
- **Reproducibility**: Fixed random seeds for deterministic results
- **Numerical stability**: Careful handling of edge cases and numerical precision
- **Scalability**: Efficient implementations suitable for datasets up to 10,000+ strains

## 1. Network Risk Scoring (Innovation)

### Purpose

Novel metric for MDR risk prediction combining network topology with statistical confidence.

### Algorithm

```
FUNCTION compute_network_mdr_risk_score(network, strain_features, bootstrap_ci):
    INPUT:
        network: Hybrid co-resistance network (NetworkX Graph)
        strain_features: Binary matrix of strain features (DataFrame)
        bootstrap_ci: Bootstrap confidence intervals (Dict)
    
    OUTPUT:
        DataFrame with (Strain_ID, Network_Risk_Score, MDR_Predicted, Percentile_Rank)
    
    PROCEDURE:
        # 1. Compute centrality metrics
        degree_cent = degree_centrality(network)
        betweenness_cent = betweenness_centrality(network)
        eigenvector_cent = eigenvector_centrality(network)
        
        # 2. Weight by bootstrap CI width (narrower = higher confidence)
        FOR EACH node in bootstrap_ci:
            ci_width = ci_upper - ci_lower
            ci_weights[node] = 1.0 / (ci_width + ε)
        
        # 3. Compute weighted score for each strain
        FOR EACH strain:
            present_features = features where value == 1
            score = 0.0
            FOR EACH feature in present_features:
                IF feature in network.nodes:
                    cent_score = (degree_cent[feature] × 0.4 +
                                 betweenness_cent[feature] × 0.3 +
                                 eigenvector_cent[feature] × 0.3)
                    score += cent_score × ci_weights[feature]
            
            risk_scores.append({Strain_ID, score})
        
        # 4. Compute percentile rank and MDR prediction
        threshold = percentile(risk_scores, 75%)
        FOR EACH strain:
            MDR_Predicted = (score > threshold)
            Percentile_Rank = rank(score) / total × 100
        
        RETURN DataFrame(risk_scores)
```

**Complexity**:
- Time: O(n × m × C) where n=strains, m=features, C=centrality computation
- Space: O(n + m) for storing scores and centralities

**Innovation**: First tool to combine network topology with bootstrap confidence for risk prediction.

---

## 2. Sequential Pattern Detection (Innovation)

### Purpose

Identify order-dependent resistance acquisition patterns (A→B→C) suggesting sequential evolution.

### Algorithm

```
FUNCTION detect_sequential_resistance_patterns(data, min_support, min_confidence):
    INPUT:
        data: Binary matrix of resistance features (DataFrame)
        min_support: Minimum pattern frequency (default: 0.1)
        min_confidence: Minimum sequential confidence (default: 0.5)
    
    OUTPUT:
        DataFrame with (Pattern, Support, Confidence, Lift, P_Value)
    
    PROCEDURE:
        # 1. Compute correlation matrix to infer order
        corr_matrix = correlation(data)
        
        # 2. Find potential sequential relationships
        sequential_edges = []
        FOR EACH pair (A, B):
            IF corr(A, B) > threshold:
                sequential_edges.append((A, B, corr(A, B)))
        
        # 3. Mine sequential patterns (length 2: A→B)
        FOR EACH edge (A, B) in sequential_edges:
            support_ab = P(A ∩ B)
            IF support_ab >= min_support:
                confidence = P(B | A) = support_ab / P(A)
                IF confidence >= min_confidence:
                    lift = confidence / P(B)
                    p_value = statistical_test_vs_random(A, B)
                    patterns.append({Pattern: "A→B", Support, Confidence, Lift, P_Value})
        
        # 4. Mine sequential patterns (length 3: A→B→C)
        FOR EACH triplet (A, B, C):
            support_abc = P(A ∩ B ∩ C)
            IF support_abc >= min_support:
                confidence = P(B|A) × P(C|B)
                IF confidence >= min_confidence:
                    patterns.append({Pattern: "A→B→C", ...})
        
        RETURN patterns sorted by confidence
```

**Complexity**:
- Time: O(m² × n + m³) where m=features, n=samples
- Space: O(m²) for correlation matrix

**Innovation**: First tool to identify order-dependent patterns in genomic resistance data.

---

## 3. Bootstrap Confidence Interval Computation

### Percentile Bootstrap (Current Implementation)

**Purpose**: Estimate confidence intervals for prevalence proportions without parametric assumptions.

**Algorithm**:
```
FUNCTION compute_bootstrap_ci(data, n_iter=5000, confidence_level=0.95):
    INPUT:
        data: Binary DataFrame with n samples and m features
        n_iter: Number of bootstrap iterations
        confidence_level: Confidence level (e.g., 0.95)
    
    OUTPUT:
        DataFrame with (ColumnName, Mean, CI_Lower, CI_Upper)
    
    PROCEDURE:
        alpha = 1.0 - confidence_level
        results = empty list
        
        FOR EACH column c in data.columns:
            col_data = data[c].values  # Array of n values
            n = len(col_data)
            
            # Generate bootstrap distribution
            boot_stats = empty array of size n_iter
            FOR i = 1 TO n_iter:
                sample = random_choice(col_data, size=n, replace=True)
                boot_stats[i] = mean(sample)
            
            # Compute percentile-based CI
            ci_lower = percentile(boot_stats, alpha/2 * 100)
            ci_upper = percentile(boot_stats, (1 - alpha/2) * 100)
            
            results.append({
                ColumnName: c,
                Mean: mean(boot_stats) * 100,
                CI_Lower: ci_lower * 100,
                CI_Upper: ci_upper * 100
            })
        
        RETURN DataFrame(results).sort_by(Mean, descending=True)
```

**Complexity**:
- Time: O(m × n_iter × n) where m = features, n = samples
- Space: O(n_iter) for storing bootstrap statistics

**Convergence Criteria**:
- Default n_iter = 5000 for production
- CI width difference between 5000 and 10000 iterations should be < 5%

### BCa Bootstrap (Advanced Implementation)

**Purpose**: Bias-corrected and accelerated (BCa) bootstrap for more accurate CIs.

**Algorithm**:
```
FUNCTION compute_bootstrap_ci_bca(data, n_iter=5000, confidence_level=0.95):
    alpha = 1.0 - confidence_level
    
    FOR EACH column in data:
        col_data = column.values
        n = len(col_data)
        theta_hat = mean(col_data)
        
        # Bootstrap distribution
        boot_stats = [mean(bootstrap_sample) for _ in range(n_iter)]
        
        # Bias correction factor z0
        n_less = count(boot_stats < theta_hat)
        z0 = norm.ppf(n_less / n_iter) if 0 < n_less < n_iter else 0.0
        
        # Acceleration factor a_hat (jackknife)
        jack_stats = [mean(delete(col_data, i)) for i in range(n)]
        jack_mean = mean(jack_stats)
        num = sum((jack_mean - jack_stats)^3)
        den = 6.0 * (sum((jack_mean - jack_stats)^2))^1.5
        a_hat = num / den if den > 0 else 0.0
        
        # BCa-adjusted percentiles
        z_alpha_2 = norm.ppf(alpha / 2)
        z_1_alpha_2 = norm.ppf(1 - alpha / 2)
        
        alpha1 = cdf(z0 + (z0 + z_alpha_2) / (1 - a_hat * (z0 + z_alpha_2)))
        alpha2 = cdf(z0 + (z0 + z_1_alpha_2) / (1 - a_hat * (z0 + z_1_alpha_2)))
        
        ci_lower = quantile(boot_stats, alpha1)
        ci_upper = quantile(boot_stats, alpha2)
```

**Complexity**:
- Time: O(m × (n_iter × n + n²)) - additional O(n²) for jackknife
- Space: O(n_iter + n)

## 2. Pairwise Co-occurrence Analysis

**Purpose**: Identify statistically significant associations between binary features.

**Algorithm**:
```
FUNCTION pairwise_cooccurrence(data, alpha=0.05, method='fdr_bh'):
    INPUT:
        data: Binary DataFrame with m features
        alpha: Significance threshold
        method: Multiple testing correction method
    
    OUTPUT:
        DataFrame with significant pairs (Item1, Item2, Phi, Raw_p, Corrected_p)
    
    PROCEDURE:
        pairs = all_combinations(data.columns, 2)
        n_pairs = m * (m-1) / 2
        
        raw_pvals = empty list
        combos = empty list
        
        FOR EACH (col1, col2) in pairs:
            table = crosstab(data[col1], data[col2])  # 2×2 contingency
            chi2, p_val, phi = safe_contingency(table)
            
            IF p_val is not NaN:
                combos.append((col1, col2, phi))
                raw_pvals.append(p_val)
        
        # Multiple testing correction (Benjamini-Hochberg)
        reject, corrected_pvals = multipletests(raw_pvals, alpha, method)
        
        # Filter significant results
        results = []
        FOR EACH combo, raw_p, corr_p, is_sig in zip(...):
            IF is_sig:
                results.append({Item1, Item2, Phi, Raw_p, Corrected_p})
        
        RETURN DataFrame(results).sort_by(Corrected_p)
```

**Complexity**:
- Time: O(m² × n) for contingency tables + O(m² log m²) for FDR correction
- Space: O(m²) for storing all p-values

## 3. Contingency Table Analysis (safe_contingency)

**Purpose**: Perform appropriate statistical test based on expected cell counts.

**Algorithm**:
```
FUNCTION safe_contingency(table):
    INPUT: 2×2 contingency table [[a, b], [c, d]]
    OUTPUT: (chi2_statistic, p_value, phi_coefficient)
    
    PROCEDURE:
        IF table.shape != (2, 2) OR total == 0:
            RETURN (NaN, NaN, NaN)
        
        # Calculate phi coefficient
        (a, b), (c, d) = table.values
        row_sums = [a+b, c+d]
        col_sums = [a+c, b+d]
        total = a + b + c + d
        
        num = a*d - b*c
        den = sqrt(row_sums[0] * row_sums[1] * col_sums[0] * col_sums[1])
        phi = num / den if den > 0 else NaN
        
        # Calculate expected counts
        expected = outer(row_sums, col_sums) / total
        min_expected = min(expected)
        pct_above_5 = count(expected >= 5) / 4
        
        # Cochran's rule for test selection
        IF min_expected < 1 OR pct_above_5 < 0.8:
            # Use Fisher's exact test
            _, p_val = fisher_exact(table)
            chi2 = phi² × total  # Derived for consistency
        ELSE:
            # Use chi-square test
            chi2, p_val = chi2_contingency(table)
        
        RETURN (chi2, p_val, phi)
```

**Complexity**:
- Time: O(1) for 2×2 tables, O(n!) worst case for Fisher's exact
- Space: O(1)

## 4. FDR Correction (Benjamini-Hochberg)

**Purpose**: Control False Discovery Rate when testing multiple hypotheses.

**Algorithm**:
```
FUNCTION fdr_bh_correction(p_values, alpha=0.05):
    INPUT:
        p_values: Array of m raw p-values
        alpha: Target FDR level
    
    OUTPUT:
        reject: Boolean array of rejection decisions
        corrected: Adjusted p-values
    
    PROCEDURE:
        m = length(p_values)
        sorted_idx = argsort(p_values)
        sorted_p = p_values[sorted_idx]
        
        # Calculate BH critical values
        critical = [(i+1) / m * alpha for i in range(m)]
        
        # Find largest k where p_k <= critical_k
        k = max([i for i in range(m) if sorted_p[i] <= critical[i]], default=-1)
        
        # Reject all hypotheses with index <= k
        reject = array of False, length m
        IF k >= 0:
            reject[sorted_idx[:k+1]] = True
        
        # Calculate adjusted p-values
        # Adjusted p_i = min(p_i * m / i, 1)
        # Ensure monotonicity: adj_p[i] >= adj_p[i-1]
        corrected = empty array of size m
        corrected[sorted_idx[-1]] = min(sorted_p[-1], 1.0)
        FOR i = m-2 TO 0:
            adj = min(sorted_p[i] * m / (i+1), corrected[sorted_idx[i+1]])
            corrected[sorted_idx[i]] = adj
        
        RETURN (reject, corrected)
```

**Mathematical Properties**:
1. Corrected p-values are monotonically non-decreasing when sorted by raw p-values
2. Corrected p-values ≥ raw p-values (within numerical tolerance 1e-10)
3. Controls FDR at level α

**Complexity**:
- Time: O(m log m) dominated by sorting
- Space: O(m)

## 5. K-modes Clustering

**Purpose**: Cluster categorical data (e.g., binary resistance profiles).

**Algorithm**:
```
FUNCTION kmodes_clustering(data, k, max_iter=100):
    INPUT:
        data: Binary DataFrame with n samples and m features
        k: Number of clusters
        max_iter: Maximum iterations
    
    OUTPUT:
        cluster_labels: Array of n cluster assignments
        centroids: Array of k mode vectors
    
    PROCEDURE:
        # Initialize centroids randomly from data points
        centroids = random_sample(data, k)
        
        FOR iter = 1 TO max_iter:
            # Assignment step: assign each point to nearest centroid
            FOR i = 1 TO n:
                distances = [hamming_distance(data[i], centroids[j]) for j in range(k)]
                cluster_labels[i] = argmin(distances)
            
            # Update step: compute new centroids as mode of each cluster
            old_centroids = copy(centroids)
            FOR j = 1 TO k:
                cluster_points = data[cluster_labels == j]
                centroids[j] = mode(cluster_points)  # Most frequent value per feature
            
            # Check convergence
            IF centroids == old_centroids:
                BREAK
        
        RETURN (cluster_labels, centroids)
```

**Cluster Selection** (Silhouette-based):
```
FUNCTION select_optimal_k(data, k_range=(2, 10)):
    best_k = 2
    best_score = -1
    
    FOR k in k_range:
        labels = kmodes_clustering(data, k)
        score = silhouette_score(data, labels, metric='hamming')
        
        IF score > best_score:
            best_score = score
            best_k = k
    
    RETURN best_k
```

**Complexity**:
- Time: O(k × n × m × max_iter)
- Space: O(k × m + n)

## 6. Louvain Community Detection

**Purpose**: Identify communities in co-occurrence networks.

**Algorithm**:
```
FUNCTION louvain_communities(graph, resolution=1.0):
    INPUT:
        graph: NetworkX graph with weighted edges
        resolution: Resolution parameter for modularity
    
    OUTPUT:
        communities: List of sets of nodes
    
    PROCEDURE:
        # Phase 1: Local optimization
        partition = {node: {node} for node in graph.nodes}
        
        REPEAT:
            improvement = False
            FOR EACH node in graph.nodes:
                current_community = partition[node]
                best_community = current_community
                best_delta_Q = 0
                
                # Try moving to neighboring communities
                FOR neighbor in graph.neighbors(node):
                    neighbor_community = partition[neighbor]
                    delta_Q = modularity_gain(node, neighbor_community)
                    
                    IF delta_Q > best_delta_Q:
                        best_delta_Q = delta_Q
                        best_community = neighbor_community
                
                IF best_community != current_community:
                    # Move node to new community
                    move(node, current_community, best_community)
                    improvement = True
        
        UNTIL NOT improvement
        
        # Phase 2: Aggregate network and repeat
        IF number_of_communities > 1:
            aggregated_graph = aggregate(graph, partition)
            sub_communities = louvain_communities(aggregated_graph)
            RETURN disaggregate(sub_communities, partition)
        ELSE:
            RETURN list(partition.values())
```

**Modularity**:
Q = (1/2m) Σ[A_ij - k_i×k_j/(2m)] δ(c_i, c_j)

Where:
- m = total edge weight
- A_ij = adjacency matrix
- k_i, k_j = node degrees
- c_i, c_j = community assignments
- δ = Kronecker delta

**Complexity**:
- Time: O(n log n) average case for sparse networks
- Space: O(n + e) where e = number of edges

## 7. Association Rule Mining (Apriori)

**Purpose**: Discover frequent itemsets and association rules.

**Algorithm**:
```
FUNCTION apriori(transactions, min_support):
    INPUT:
        transactions: Binary DataFrame (each row is a transaction)
        min_support: Minimum support threshold
    
    OUTPUT:
        frequent_itemsets: DataFrame with itemset and support
    
    PROCEDURE:
        n = number of transactions
        
        # Generate 1-itemsets
        L1 = {item: count(item)/n for item in all_items if count(item)/n >= min_support}
        
        k = 1
        L = {1: L1}
        
        WHILE L[k] is not empty:
            # Generate candidate (k+1)-itemsets
            C_k1 = generate_candidates(L[k])
            
            # Count support for candidates
            FOR EACH transaction t:
                FOR EACH candidate c in C_k1:
                    IF c ⊆ t:
                        count[c] += 1
            
            # Filter by minimum support
            L[k+1] = {c: count[c]/n for c in C_k1 if count[c]/n >= min_support}
            k += 1
        
        RETURN union of all L[k]

FUNCTION generate_association_rules(itemsets, min_confidence, min_lift):
    rules = []
    FOR EACH itemset I with |I| >= 2:
        FOR EACH subset A of I:
            B = I - A
            confidence = support(I) / support(A)
            lift = confidence / support(B)
            
            IF confidence >= min_confidence AND lift >= min_lift:
                rules.append(A → B, support(I), confidence, lift)
    
    RETURN rules
```

**Metrics**:
- Support(A→B) = P(A ∩ B)
- Confidence(A→B) = P(B|A) = Support(A∪B) / Support(A)
- Lift(A→B) = P(A ∩ B) / (P(A) × P(B))

**Complexity**:
- Time: O(2^m) worst case, but pruning via Apriori property significantly reduces this
- Space: O(2^m) worst case for storing itemsets

## 8. Network Construction

**Purpose**: Build co-occurrence networks from significant associations.

**Algorithm**:
```
FUNCTION build_co_resistance_network(data, pheno_cols, gene_cols, alpha=0.05):
    INPUT:
        data: Combined phenotype/genotype DataFrame
        pheno_cols, gene_cols: Column name lists
        alpha: Significance threshold
    
    OUTPUT:
        G: NetworkX graph
    
    PROCEDURE:
        all_cols = pheno_cols + gene_cols
        G = empty Graph()
        
        # Add nodes
        FOR p in pheno_cols:
            G.add_node(p, node_type='Phenotype')
        FOR g in gene_cols:
            G.add_node(g, node_type='Gene')
        
        # Calculate all pairwise associations
        combos = []
        FOR (c1, c2) in combinations(all_cols, 2):
            table = crosstab(data[c1], data[c2])
            chi2, p_val, phi = safe_contingency(table)
            IF p_val is not NaN:
                combos.append((c1, c2, phi, p_val))
        
        # Apply FDR correction
        raw_pvals = [c[3] for c in combos]
        reject, corrected = multipletests(raw_pvals, alpha, 'fdr_bh')
        
        # Add significant edges
        FOR (c1, c2, phi, p_raw), p_corr, is_sig in zip(combos, corrected, reject):
            IF is_sig:
                edge_type = determine_edge_type(c1, c2, pheno_cols, gene_cols)
                G.add_edge(c1, c2, phi=phi, pvalue=p_corr, edge_type=edge_type)
        
        # Remove isolated nodes
        G.remove_nodes(list(isolates(G)))
        
        RETURN G
```

**Complexity**:
- Time: O(m² × n) for contingency tables + O(m² log m²) for FDR
- Space: O(m² + n_edges)

---

## Scalability Considerations

### Typical Performance

| Operation | 100 strains | 500 strains | 1000 strains |
|-----------|-------------|-------------|--------------|
| Bootstrap CI (5000 iter, 20 features) | ~2s | ~5s | ~10s |
| Pairwise co-occurrence (50 features) | ~3s | ~8s | ~15s |
| K-modes clustering | ~1s | ~5s | ~12s |
| Full pipeline | ~30s | ~90s | ~180s |

### Memory Optimization

For large datasets (>1000 strains):
1. Use chunked processing for bootstrap
2. Sparse matrix representation for binary data
3. Generator-based pairwise iteration

### Parallelization

Current implementation uses:
- `ProcessPoolExecutor` for bootstrap column parallelization
- Potential for GPU acceleration in future versions

---

## References

1. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
2. Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS-B*, 57(1), 289-300.
3. Huang, Z. (1998). Extensions to the k-means algorithm for clustering large data sets. *Data Mining and Knowledge Discovery*, 2(3), 283-304.
4. Blondel, V. D., et al. (2008). Fast unfolding of communities in large networks. *JSTAT*, 2008(10), P10008.
5. Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules. *VLDB*, 1215, 487-499.

---

**Version**: 1.0.0  
**Last Updated**: 2025-01-15
