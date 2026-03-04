{\rtf1\ansi\ansicpg1252\cocoartf2867
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 Menlo-Bold;\f1\fnil\fcharset0 Menlo-Regular;\f2\fnil\fcharset0 Menlo-Italic;
}
{\colortbl;\red255\green255\blue255;\red70\green137\blue204;\red24\green24\blue24;\red193\green193\blue193;
\red202\green202\blue202;\red212\green214\blue154;\red167\green197\blue152;\red85\green129\blue224;\red194\green126\blue101;
}
{\*\expandedcolortbl;;\cssrgb\c33725\c61176\c83922;\cssrgb\c12157\c12157\c12157;\cssrgb\c80000\c80000\c80000;
\cssrgb\c83137\c83137\c83137;\cssrgb\c86275\c86275\c66667;\cssrgb\c70980\c80784\c65882;\cssrgb\c40392\c58824\c90196;\cssrgb\c80784\c56863\c47059;
}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\b\fs24 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 # Context Document: SPI-SPI Feature Space for Comparing Multivariate Time Series of Arbitrary Dimension
\f1\b0 \cf4 \cb1 \strokec4 \
\

\f0\b \cf2 \cb3 \strokec2 ## Purpose of This Document
\f1\b0 \cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 This document provides complete methodological context for ongoing work toward an EEML 2026 extended abstract submission. It describes a method for embedding multivariate time series (MTS) of arbitrary spatial dimension $\cf5 \strokec5 M\cf4 \strokec4 $ and temporal length $\cf5 \strokec5 T\cf4 \strokec4 $ into a common feature space. This document covers: (1) the problem and motivation, (2) the baseline method (SPI-SPI space), (3) its known limitations, and (4) illustrative case studies. A companion document describes the proposed extension using geometric deep learning.\cb1 \
\
\cb3 ---\cb1 \
\
\pard\pardeftab720\partightenfactor0

\f0\b \cf2 \cb3 \strokec2 ## 1. Problem Statement
\f1\b0 \cf4 \cb1 \strokec4 \
\

\f0\b \cf2 \cb3 \strokec2 ### 1.1 Feature-Based Univariate Time Series Analysis
\f1\b0 \cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 Feature-based time series analysis transforms a univariate time series $\cf5 \strokec5 x \cf2 \strokec2 \\in\cf5 \strokec5  \cf2 \strokec2 \\\cf6 \strokec6 mathbb\cf5 \strokec5 \{R\}^T\cf4 \strokec4 $ into a fixed-dimensional feature vector $\cf2 \strokec2 \\\cf6 \strokec6 mathbf\cf5 \strokec5 \{f\} \cf2 \strokec2 \\in\cf5 \strokec5  \cf2 \strokec2 \\\cf6 \strokec6 mathbb\cf5 \strokec5 \{R\}^K\cf4 \strokec4 $ by computing $\cf5 \strokec5 K\cf4 \strokec4 $ summary statistics (e.g., mean, autocorrelation, spectral entropy). This eliminates dependence on $\cf5 \strokec5 T\cf4 \strokec4 $, enabling comparison of time series of different lengths in a common $\cf5 \strokec5 K\cf4 \strokec4 $-dimensional feature space amenable to standard statistical and machine learning methods.\cb1 \
\
\pard\pardeftab720\partightenfactor0

\f0\b \cf2 \cb3 \strokec2 ### 1.2 The Multivariate Extension Problem
\f1\b0 \cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 For an $\cf5 \strokec5 M\cf4 \strokec4 $-channelled multivariate time series $\cf2 \strokec2 \\\cf6 \strokec6 mathbf\cf5 \strokec5 \{X\} \cf2 \strokec2 \\in\cf5 \strokec5  \cf2 \strokec2 \\\cf6 \strokec6 mathbb\cf5 \strokec5 \{R\}^\{M \cf2 \strokec2 \\times\cf5 \strokec5  T\}\cf4 \strokec4 $, the spatial dimension $\cf5 \strokec5 M\cf4 \strokec4 $ introduces a second axis of variation. Feature-based methods that summarise pairwise interactions between channels (producing, e.g., $\cf5 \strokec5 M \cf2 \strokec2 \\times\cf5 \strokec5  M\cf4 \strokec4 $ matrices) eliminate dependence on $\cf5 \strokec5 T\cf4 \strokec4 $ but remain sensitive to $\cf5 \strokec5 M\cf4 \strokec4 $. Specifically: if two MTS processes $\cf2 \strokec2 \\\cf6 \strokec6 mathbf\cf5 \strokec5 \{X\} \cf2 \strokec2 \\in\cf5 \strokec5  \cf2 \strokec2 \\\cf6 \strokec6 mathbb\cf5 \strokec5 \{R\}^\{M_\cf7 \strokec7 1\cf5 \strokec5  \cf2 \strokec2 \\times\cf5 \strokec5  T_\cf7 \strokec7 1\cf5 \strokec5 \}\cf4 \strokec4 $ and $\cf2 \strokec2 \\\cf6 \strokec6 mathbf\cf5 \strokec5 \{Y\} \cf2 \strokec2 \\in\cf5 \strokec5  \cf2 \strokec2 \\\cf6 \strokec6 mathbb\cf5 \strokec5 \{R\}^\{M_\cf7 \strokec7 2\cf5 \strokec5  \cf2 \strokec2 \\times\cf5 \strokec5  T_\cf7 \strokec7 2\cf5 \strokec5 \}\cf4 \strokec4 $ have $\cf5 \strokec5 M_\cf7 \strokec7 1\cf5 \strokec5  \cf2 \strokec2 \\neq\cf5 \strokec5  M_\cf7 \strokec7 2\cf4 \strokec4 $, their pairwise interaction matrices live in different-dimensional spaces and cannot be directly compared.\cb1 \
\
\cb3 Na\'efve solutions such as truncating to the minimum common channel count discard channels and, with them, potentially important dynamical regimes and inter-channel dependency structures. The problem is therefore: 
\f0\b \cf2 \strokec2 **how can we embed MTS of arbitrary $\cf5 \strokec5 M\cf2 \strokec2 $ and $\cf5 \strokec5 T\cf2 \strokec2 $ into a common feature space that preserves informative dependency structures?**
\f1\b0 \cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0

\f0\b \cf2 \cb3 \strokec2 ### 1.3 Existing Approaches
\f1\b0 \cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 This problem is not entirely unaddressed. Several families of methods can compare graphs (or systems) of different sizes:\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf8 \cb3 \strokec8 -\cf4 \strokec4  
\f0\b \cf2 \strokec2 **Graph kernels**
\f1\b0 \cf4 \strokec4  (Weisfeiler-Leman subtree, random walk, graphlet kernels) define kernel functions between graphs of arbitrary size and are well-established in graph classification.\cb1 \
\cf8 \cb3 \strokec8 -\cf4 \strokec4  
\f0\b \cf2 \strokec2 **Global graph pooling in GNNs**
\f1\b0 \cf4 \strokec4  (sum/mean/attention readout) produces fixed-dimensional embeddings regardless of node count.\cb1 \
\cf8 \cb3 \strokec8 -\cf4 \strokec4  
\f0\b \cf2 \strokec2 **NetSimile and similar feature-extraction approaches**
\f1\b0 \cf4 \strokec4  compute distributions of local structural features (degree, clustering coefficient, etc.) and compare their statistics.\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 The method described here is a novel instantiation within this broader family, distinguished by: (a) using a diverse, interpretable library of pairwise interaction statistics as the input representation, and (b) exploiting the 
\f2\i *relationships between*
\f1\i0  different statistical characterisations as the embedding, rather than structural graph features.\cb1 \
\
\cb3 ---\cb1 \
\
\pard\pardeftab720\partightenfactor0

\f0\b \cf2 \cb3 \strokec2 ## 2. The SPI-SPI Feature Space (Baseline Method)
\f1\b0 \cf4 \cb1 \strokec4 \
\

\f0\b \cf2 \cb3 \strokec2 ### 2.1 Pairwise Interaction Matrices (MPIs)
\f1\b0 \cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 Let $\cf2 \strokec2 \\\cf6 \strokec6 mathbf\cf5 \strokec5 \{X\} \cf2 \strokec2 \\in\cf5 \strokec5  \cf2 \strokec2 \\\cf6 \strokec6 mathbb\cf5 \strokec5 \{R\}^\{M \cf2 \strokec2 \\times\cf5 \strokec5  T\}\cf4 \strokec4 $ be an MTS with channels $\cf5 \strokec5 \\\{X_t^\{(i)\}\\\}_\{i=\cf7 \strokec7 1\cf5 \strokec5 \}^M\cf4 \strokec4 $. The Python library \cf9 \strokec9 `pyspi`\cf4 \strokec4  computes a library of $\cf5 \strokec5 K > \cf7 \strokec7 250\cf4 \strokec4 $ statistics of pairwise interaction (SPIs) between channels. For the $\cf5 \strokec5 k\cf4 \strokec4 $-th SPI, computing it on every ordered pair $\cf5 \strokec5 (i, j)\cf4 \strokec4 $ yields an $\cf5 \strokec5 M \cf2 \strokec2 \\times\cf5 \strokec5  M\cf4 \strokec4 $ adjacency matrix\cb1 \
\
\cb3 $$\cf2 \strokec2 \\\cf6 \strokec6 mathrm\cf5 \strokec5 \{MPI\}_k \cf2 \strokec2 \\in\cf5 \strokec5  \cf2 \strokec2 \\\cf6 \strokec6 mathbb\cf5 \strokec5 \{R\}^\{M \cf2 \strokec2 \\times\cf5 \strokec5  M\}, \cf2 \strokec2 \\quad\cf5 \strokec5  [\cf2 \strokec2 \\\cf6 \strokec6 mathrm\cf5 \strokec5 \{MPI\}_k]_\{ij\} = \cf2 \strokec2 \\\cf6 \strokec6 mathrm\cf5 \strokec5 \{SPI\}_k(X^\{(i)\}, X^\{(j)\}).\cf4 \strokec4 $$\cb1 \
\
\cb3 We call this a 
\f0\b \cf2 \strokec2 **matrix of pairwise interaction (MPI)**
\f1\b0 \cf4 \strokec4 . Geometrically, this is a weighted directed graph on $\cf5 \strokec5 M\cf4 \strokec4 $ nodes, where node $\cf5 \strokec5 i\cf4 \strokec4 $ represents channel $\cf5 \strokec5 X^\{(i)\}\cf4 \strokec4 $ and edge weight $\cf5 \strokec5 (i,j)\cf4 \strokec4 $ is the value of the $\cf5 \strokec5 k\cf4 \strokec4 $-th SPI computed between channels $\cf5 \strokec5 i\cf4 \strokec4 $ and $\cf5 \strokec5 j\cf4 \strokec4 $.\cb1 \
\
\cb3 Computing all $\cf5 \strokec5 K\cf4 \strokec4 $ SPIs yields a tensor $\cf2 \strokec2 \\\cf6 \strokec6 mathcal\cf5 \strokec5 \{M\} \cf2 \strokec2 \\in\cf5 \strokec5  \cf2 \strokec2 \\\cf6 \strokec6 mathbb\cf5 \strokec5 \{R\}^\{K \cf2 \strokec2 \\times\cf5 \strokec5  M \cf2 \strokec2 \\times\cf5 \strokec5  M\}\cf4 \strokec4 $ \'97 a collection of $\cf5 \strokec5 K\cf4 \strokec4 $ interaction graphs on the same $\cf5 \strokec5 M\cf4 \strokec4 $ nodes.\cb1 \
\
\pard\pardeftab720\partightenfactor0

\f0\b \cf2 \cb3 \strokec2 **Note on temporal invariance.**
\f1\b0 \cf4 \strokec4  Each SPI maps a pair of time series (of any length $\cf5 \strokec5 T\cf4 \strokec4 $) to a scalar. Therefore, $\cf2 \strokec2 \\\cf6 \strokec6 mathcal\cf5 \strokec5 \{M\}\cf4 \strokec4 $ is invariant to $\cf5 \strokec5 T\cf4 \strokec4 $. Two MTS $\cf2 \strokec2 \\\cf6 \strokec6 mathbf\cf5 \strokec5 \{X\} \cf2 \strokec2 \\in\cf5 \strokec5  \cf2 \strokec2 \\\cf6 \strokec6 mathbb\cf5 \strokec5 \{R\}^\{M \cf2 \strokec2 \\times\cf5 \strokec5  T_\cf7 \strokec7 1\cf5 \strokec5 \}\cf4 \strokec4 $ and $\cf2 \strokec2 \\\cf6 \strokec6 mathbf\cf5 \strokec5 \{Y\} \cf2 \strokec2 \\in\cf5 \strokec5  \cf2 \strokec2 \\\cf6 \strokec6 mathbb\cf5 \strokec5 \{R\}^\{M \cf2 \strokec2 \\times\cf5 \strokec5  T_\cf7 \strokec7 2\cf5 \strokec5 \}\cf4 \strokec4 $ with $\cf5 \strokec5 T_\cf7 \strokec7 1\cf5 \strokec5  \cf2 \strokec2 \\neq\cf5 \strokec5  T_\cf7 \strokec7 2\cf4 \strokec4 $ produce MPIs of identical dimension, enabling direct comparison of their interaction structures.\cb1 \
\

\f0\b \cf2 \cb3 \strokec2 **Remaining limitation.**
\f1\b0 \cf4 \strokec4  The tensor $\cf2 \strokec2 \\\cf6 \strokec6 mathcal\cf5 \strokec5 \{M\} \cf2 \strokec2 \\in\cf5 \strokec5  \cf2 \strokec2 \\\cf6 \strokec6 mathbb\cf5 \strokec5 \{R\}^\{K \cf2 \strokec2 \\times\cf5 \strokec5  M \cf2 \strokec2 \\times\cf5 \strokec5  M\}\cf4 \strokec4 $ still depends on $\cf5 \strokec5 M\cf4 \strokec4 $. Systems with different channel counts produce tensors of different size.\cb1 \
\

\f0\b \cf2 \cb3 \strokec2 ### 2.2 Constructing the SPI-SPI Feature Vector
\f1\b0 \cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 To eliminate dependence on $\cf5 \strokec5 M\cf4 \strokec4 $, we compute a second-order summary: the correlation between pairs of SPIs across all pairwise channel interactions within a single MTS.\cb1 \
\
\cb3 For the $\cf5 \strokec5 k\cf4 \strokec4 $-th MPI, extract the $\cf2 \strokec2 \\\cf6 \strokec6 binom\cf5 \strokec5 \{M\}\{\cf7 \strokec7 2\cf5 \strokec5 \}\cf4 \strokec4 $ off-diagonal entries (or all $\cf5 \strokec5 M(M-\cf7 \strokec7 1\cf5 \strokec5 )\cf4 \strokec4 $ entries if the SPI is asymmetric) into a vector $\cf2 \strokec2 \\\cf6 \strokec6 mathbf\cf5 \strokec5 \{v\}_k \cf2 \strokec2 \\in\cf5 \strokec5  \cf2 \strokec2 \\\cf6 \strokec6 mathbb\cf5 \strokec5 \{R\}^\{M(M-\cf7 \strokec7 1\cf5 \strokec5 )/\cf7 \strokec7 2\cf5 \strokec5 \}\cf4 \strokec4 $. For two SPIs $\cf5 \strokec5 k\cf4 \strokec4 $ and $\cf5 \strokec5 k'\cf4 \strokec4 $, compute\cb1 \
\
\cb3 $$\cf5 \strokec5 f_\{kk'\} = \cf2 \strokec2 \\\cf6 \strokec6 mathrm\cf5 \strokec5 \{corr\}(\cf2 \strokec2 \\\cf6 \strokec6 mathbf\cf5 \strokec5 \{v\}_k, \cf2 \strokec2 \\\cf6 \strokec6 mathbf\cf5 \strokec5 \{v\}_\{k'\})\cf4 \strokec4 $$\cb1 \
\
\cb3 where $\cf2 \strokec2 \\\cf6 \strokec6 mathrm\cf5 \strokec5 \{corr\}\cf4 \strokec4 $ is Pearson's product-moment correlation. This yields a scalar summarising how similarly the two SPIs rank the pairwise interactions within the system.\cb1 \
\
\cb3 For $\cf5 \strokec5 K\cf4 \strokec4 $ SPIs, we obtain a $\cf2 \strokec2 \\\cf6 \strokec6 binom\cf5 \strokec5 \{K\}\{\cf7 \strokec7 2\cf5 \strokec5 \}\cf4 \strokec4 $-dimensional feature vector\cb1 \
\
\cb3 $$\cf2 \strokec2 \\\cf6 \strokec6 mathbf\cf5 \strokec5 \{f\} = \\\{f_\{kk'\}\\\}_\{k < k'\} \cf2 \strokec2 \\in\cf5 \strokec5  \cf2 \strokec2 \\\cf6 \strokec6 mathbb\cf5 \strokec5 \{R\}^\{\cf2 \strokec2 \\\cf6 \strokec6 binom\cf5 \strokec5 \{K\}\{\cf7 \strokec7 2\cf5 \strokec5 \}\}\cf4 \strokec4 $$\cb1 \
\
\cb3 which we term the 
\f0\b \cf2 \strokec2 **SPI-SPI feature vector**
\f1\b0 \cf4 \strokec4 , and the resulting space 
\f0\b \cf2 \strokec2 **SPI-SPI space**
\f1\b0 \cf4 \strokec4 .\cb1 \
\
\pard\pardeftab720\partightenfactor0

\f0\b \cf2 \cb3 \strokec2 **Key property.**
\f1\b0 \cf4 \strokec4  The dimension of $\cf2 \strokec2 \\\cf6 \strokec6 mathbf\cf5 \strokec5 \{f\}\cf4 \strokec4 $ depends only on $\cf5 \strokec5 K\cf4 \strokec4 $ (the number of SPIs chosen), not on $\cf5 \strokec5 M\cf4 \strokec4 $ or $\cf5 \strokec5 T\cf4 \strokec4 $. Therefore, any two MTS \'97 regardless of their spatial or temporal dimensions \'97 can be embedded into this common $\cf2 \strokec2 \\\cf6 \strokec6 binom\cf5 \strokec5 \{K\}\{\cf7 \strokec7 2\cf5 \strokec5 \}\cf4 \strokec4 $-dimensional space.\cb1 \
\

\f0\b \cf2 \cb3 \strokec2 ### 2.3 Interpretation
\f1\b0 \cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 Each feature $\cf5 \strokec5 f_\{kk'\}\cf4 \strokec4 $ encodes the empirical relationship between two statistical characterisations of pairwise interaction across the channels of a single system. Comparing $\cf2 \strokec2 \\\cf6 \strokec6 mathbf\cf5 \strokec5 \{f\}\cf4 \strokec4 $ across different MTS processes reveals similarities and differences in the 
\f0\b \cf2 \strokec2 **character**
\f1\b0 \cf4 \strokec4  or 
\f0\b \cf2 \strokec2 **nature**
\f1\b0 \cf4 \strokec4  of dependency structures, rather than in the strength of specific channel-pair interactions.\cb1 \
\
\cb3 This is a critical distinction: SPI-SPI space is agnostic to 
\f2\i *which*
\f1\i0  channels are coupled. It captures the statistical fingerprint of the coupling regime but discards the topological arrangement. This is both a strength (enabling comparison across different $\cf5 \strokec5 M\cf4 \strokec4 $) and a limitation (see \'a72.4).\cb1 \
\
\pard\pardeftab720\partightenfactor0

\f0\b \cf2 \cb3 \strokec2 ### 2.4 Choice of Correlation Measure
\f1\b0 \cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 The off-diagonal entries of an MPI are not statistically independent \'97 entries sharing a node (e.g., $\cf2 \strokec2 \\\cf6 \strokec6 mathrm\cf5 \strokec5 \{SPI\}_k(\cf7 \strokec7 1\cf5 \strokec5 ,\cf7 \strokec7 2\cf5 \strokec5 )\cf4 \strokec4 $ and $\cf2 \strokec2 \\\cf6 \strokec6 mathrm\cf5 \strokec5 \{SPI\}_k(\cf7 \strokec7 1\cf5 \strokec5 ,\cf7 \strokec7 3\cf5 \strokec5 )\cf4 \strokec4 $) are confounded by the marginal properties of the shared channel. This 
\f0\b \cf2 \strokec2 **network autocorrelation**
\f1\b0 \cf4 \strokec4  inflates the effective sample size when computing $\cf2 \strokec2 \\\cf6 \strokec6 mathrm\cf5 \strokec5 \{corr\}(\cf2 \strokec2 \\\cf6 \strokec6 mathbf\cf5 \strokec5 \{v\}_k, \cf2 \strokec2 \\\cf6 \strokec6 mathbf\cf5 \strokec5 \{v\}_\{k'\})\cf4 \strokec4 $, potentially leading to overconfident correlation estimates.\cb1 \
\
\cb3 We use Pearson correlation rather than Spearman rank correlation for $\cf5 \strokec5 f_\{kk'\}\cf4 \strokec4 $. The rationale: rank transformation may corrupt subtle quantitative signatures \'97 specifically, when the signal lies in the relative 
\f2\i *magnitudes*
\f1\i0  of SPI values across channel pairs (not just their ordering), Spearman's rank space discards this information. Since the method's discriminative power relies on detecting fine-grained shifts in the covariation of SPIs, Pearson is preferred.\cb1 \
\
\cb3 However, this choice should be validated empirically. A robustness check comparing Pearson, Spearman, and Kendall's $\cf2 \strokec2 \\tau\cf4 \strokec4 $ for the inter-MPI correlation is warranted. Additionally, permutation-based significance testing or QAP (quadratic assignment procedure, standard in social network analysis) should be considered to account for the non-independence structure.\cb1 \
\
\pard\pardeftab720\partightenfactor0

\f0\b \cf2 \cb3 \strokec2 ### 2.5 Known Limitations
\f1\b0 \cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf8 \cb3 \strokec8 1.\cf4 \strokec4  
\f0\b \cf2 \strokec2 **Information bottleneck.**
\f1\b0 \cf4 \strokec4  Reducing an $\cf5 \strokec5 M \cf2 \strokec2 \\times\cf5 \strokec5  M\cf4 \strokec4 $ adjacency matrix to a single scalar per SPI pair is aggressive. Graph topology \'97 community structure, hub-spoke patterns, motifs, degree distribution \'97 is entirely discarded. Two systems with identical coupling 
\f2\i *character*
\f1\i0  but different network architectures (e.g., star vs. ring topology) are indistinguishable in SPI-SPI space.\cb1 \
\
\cf8 \cb3 \strokec8 2.\cf4 \strokec4  
\f0\b \cf2 \strokec2 **Scalability of feature space.**
\f1\b0 \cf4 \strokec4  For $\cf5 \strokec5 K\cf4 \strokec4 $ SPIs, the feature vector has $\cf2 \strokec2 \\\cf6 \strokec6 binom\cf5 \strokec5 \{K\}\{\cf7 \strokec7 2\cf5 \strokec5 \}\cf4 \strokec4 $ entries. With $\cf5 \strokec5 K = \cf7 \strokec7 250\cf4 \strokec4 $, this is 31,125 features \'97 high-dimensional relative to typical sample sizes, requiring careful regularisation or feature selection.\cb1 \
\
\cf8 \cb3 \strokec8 3.\cf4 \strokec4  
\f0\b \cf2 \strokec2 **No learned representation.**
\f1\b0 \cf4 \strokec4  The features are hand-crafted. There is no mechanism to learn which SPIs or which inter-SPI relationships are most informative for a given task.\cb1 \
\
\cf8 \cb3 \strokec8 4.\cf4 \strokec4  
\f0\b \cf2 \strokec2 **Network autocorrelation in the correlation estimate.**
\f1\b0 \cf4 \strokec4  As noted in \'a72.4, the non-independence of off-diagonal entries biases the correlation. This is a statistical subtlety that does not invalidate the method but requires careful treatment.\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 ---\cb1 \
\
\pard\pardeftab720\partightenfactor0

\f0\b \cf2 \cb3 \strokec2 ## 3. Illustrative Case Studies
\f1\b0 \cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 These case studies demonstrate the interpretive logic of SPI-SPI space on controlled synthetic systems. They serve as pedagogical motivation, not as the primary contribution.\cb1 \
\
\pard\pardeftab720\partightenfactor0

\f0\b \cf2 \cb3 \strokec2 ### 3.1 Differentiating Linear, Nonlinear, and Non-Monotonic Dependencies
\f1\b0 \cf4 \cb1 \strokec4 \
\

\f0\b \cf2 \cb3 \strokec2 **Setup.**
\f1\b0 \cf4 \strokec4  We construct an $\cf5 \strokec5 M \cf2 \strokec2 \\times\cf5 \strokec5  T\cf4 \strokec4 $ MTS where each channel is a noisy, filtered copy of a latent autoregressive process\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 $$\cf5 \strokec5 z_t = a \cf2 \strokec2 \\cdot\cf5 \strokec5  z_\{t-\cf7 \strokec7 1\cf5 \strokec5 \} + \cf2 \strokec2 \\epsilon_t\cf5 \strokec5 , \cf2 \strokec2 \\quad\cf5 \strokec5  \cf2 \strokec2 \\epsilon_t\cf5 \strokec5  \cf2 \strokec2 \\sim\cf5 \strokec5  \cf2 \strokec2 \\\cf6 \strokec6 mathcal\cf5 \strokec5 \{N\}(\cf7 \strokec7 0\cf5 \strokec5 , \cf7 \strokec7 1\cf5 \strokec5 ),\cf4 \strokec4 $$\cb1 \
\
\cb3 passed through a nonlinear filter $\cf5 \strokec5 g(z; \cf2 \strokec2 \\alpha\cf5 \strokec5 )\cf4 \strokec4 $ and corrupted by observation noise:\cb1 \
\
\cb3 $$\cf5 \strokec5 X_t^\{(i)\} = g(z_t; \cf2 \strokec2 \\alpha_i\cf5 \strokec5 ) + \cf2 \strokec2 \\eta_t\cf5 \strokec5 ^\{(i)\}, \cf2 \strokec2 \\quad\cf5 \strokec5  \cf2 \strokec2 \\eta_t\cf5 \strokec5 ^\{(i)\} \cf2 \strokec2 \\sim\cf5 \strokec5  \cf2 \strokec2 \\\cf6 \strokec6 mathcal\cf5 \strokec5 \{N\}(\cf7 \strokec7 0\cf5 \strokec5 , \cf2 \strokec2 \\sigma_\\eta\cf5 \strokec5 ^\cf7 \strokec7 2\cf5 \strokec5 ).\cf4 \strokec4 $$\cb1 \
\
\cb3 The filter $\cf5 \strokec5 g(z; \cf2 \strokec2 \\alpha\cf5 \strokec5 )\cf4 \strokec4 $ is defined as: (1) rescale $\cf5 \strokec5 z_t\cf4 \strokec4 $ to $\cf5 \strokec5 [\cf7 \strokec7 0\cf5 \strokec5 , \cf7 \strokec7 1\cf5 \strokec5 ]\cf4 \strokec4 $ via min-max normalisation, (2) map to the interval $\cf5 \strokec5 [-\cf2 \strokec2 \\alpha\cf5 \strokec5 , \cf2 \strokec2 \\alpha\cf5 \strokec5 ]\cf4 \strokec4 $, (3) apply $\cf2 \strokec2 \\sin\cf5 \strokec5 (\cf2 \strokec2 \\cdot\cf5 \strokec5 )\cf4 \strokec4 $, and (4) normalise to $\cf5 \strokec5 [-\cf7 \strokec7 1\cf5 \strokec5 , \cf7 \strokec7 1\cf5 \strokec5 ]\cf4 \strokec4 $ by dividing by the theoretical maximum $\cf5 \strokec5 g_\{\cf2 \strokec2 \\max\cf5 \strokec5 \} = \cf2 \strokec2 \\min\cf5 \strokec5 (\cf2 \strokec2 \\sin\cf5 \strokec5 (\cf2 \strokec2 \\alpha\cf5 \strokec5 ), \cf7 \strokec7 1\cf5 \strokec5 )\cf4 \strokec4 $. The normalisation preserves a constant signal-to-noise ratio across filter configurations.\cb1 \
\
\pard\pardeftab720\partightenfactor0

\f0\b \cf2 \cb3 \strokec2 **Experimental logic.**
\f1\b0 \cf4 \strokec4  The parameter $\cf2 \strokec2 \\alpha\cf4 \strokec4 $ controls the degree of nonlinearity:\cb1 \
\pard\pardeftab720\partightenfactor0
\cf8 \cb3 \strokec8 -\cf4 \strokec4  For $\cf2 \strokec2 \\alpha\cf5 \strokec5  \cf2 \strokec2 \\ll\cf5 \strokec5  \cf7 \strokec7 1\cf4 \strokec4 $: $\cf2 \strokec2 \\sin\cf5 \strokec5 (x) \cf2 \strokec2 \\approx\cf5 \strokec5  x\cf4 \strokec4 $ (linear regime).\cb1 \
\cf8 \cb3 \strokec8 -\cf4 \strokec4  For $\cf2 \strokec2 \\alpha\cf4 \strokec4 $ approaching $\cf2 \strokec2 \\pi\cf5 \strokec5 /\cf7 \strokec7 2\cf4 \strokec4 $: the mapping becomes nonlinear but monotonic.\cb1 \
\cf8 \cb3 \strokec8 -\cf4 \strokec4  For $\cf2 \strokec2 \\alpha\cf5 \strokec5  > \cf2 \strokec2 \\pi\cf5 \strokec5 /\cf7 \strokec7 2\cf4 \strokec4 $: the mapping becomes non-monotonic.\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 Each channel $\cf5 \strokec5 i\cf4 \strokec4 $ draws $\cf2 \strokec2 \\alpha_i\cf5 \strokec5  \cf2 \strokec2 \\sim\cf5 \strokec5  \cf2 \strokec2 \\\cf6 \strokec6 mathrm\cf5 \strokec5 \{Uniform\}(\cf2 \strokec2 \\pi\cf5 \strokec5 /\cf2 \strokec2 \\epsilon\cf5 \strokec5 , A)\cf4 \strokec4 $ where $\cf2 \strokec2 \\epsilon\cf4 \strokec4 $ is small (e.g., $\cf2 \strokec2 \\epsilon\cf5 \strokec5  = \cf7 \strokec7 1\cf5 \strokec5 /\cf7 \strokec7 64\cf4 \strokec4 $). The control parameter $\cf5 \strokec5 A\cf4 \strokec4 $ governs the maximum nonlinearity across channels. As $\cf5 \strokec5 A\cf4 \strokec4 $ increases toward $\cf2 \strokec2 \\pi\cf4 \strokec4 $, a greater proportion of channels are filtered through nonlinear (and eventually non-monotonic) transformations.\cb1 \
\
\pard\pardeftab720\partightenfactor0

\f0\b \cf2 \cb3 \strokec2 **Expected behaviour of $\cf5 \strokec5 f_\{kk'\} = \cf2 \strokec2 \\\cf6 \strokec6 mathrm\cf5 \strokec5 \{corr\}(r, \cf2 \strokec2 \\rho\cf5 \strokec5 )\cf2 \strokec2 $.**
\f1\b0 \cf4 \strokec4  In the linear regime, Pearson's $\cf5 \strokec5 r\cf4 \strokec4 $ and Spearman's $\cf2 \strokec2 \\rho\cf4 \strokec4 $ agree closely across channel pairs, yielding $\cf5 \strokec5 f_\{kk'\} \cf2 \strokec2 \\approx\cf5 \strokec5  \cf7 \strokec7 1\cf4 \strokec4 $. As more channels undergo nonlinear filtering, the Pearson MPI values degrade (violating linearity/Gaussianity assumptions) while Spearman values remain stable (rank-invariance under monotonic transformation). The result is a progressive decoupling: $\cf5 \strokec5 f_\{kk'\}\cf4 \strokec4 $ decreases as a function of $\cf5 \strokec5 A\cf4 \strokec4 $, with a sharper transition near $\cf5 \strokec5 A = \cf2 \strokec2 \\pi\cf5 \strokec5 /\cf7 \strokec7 2\cf4 \strokec4 $ where non-monotonicity begins to also degrade Spearman.\cb1 \
\

\f0\b \cf2 \cb3 \strokec2 **Interpretation.**
\f1\b0 \cf4 \strokec4  The degree to which $\cf5 \strokec5 r\cf4 \strokec4 $ and $\cf2 \strokec2 \\rho\cf4 \strokec4 $ are correlated across channel pairs (the feature $\cf5 \strokec5 f_\{kk'\}\cf4 \strokec4 $) provides a signature of the proportion and character of nonlinear interactions in the system. This illustrates how SPI-SPI features encode dependency 
\f2\i *character*
\f1\i0  without reference to specific channel identities.\cb1 \
\

\f0\b \cf2 \cb3 \strokec2 **Important caveat.**
\f1\b0 \cf4 \strokec4  The latent driver $\cf5 \strokec5 z_t\cf4 \strokec4 $ is itself a non-monotonic autoregressive process. The method characterises the nature of 
\f2\i *inter-channel*
\f1\i0  dependence, not the marginal dynamics of individual channels. The filter $\cf5 \strokec5 g\cf4 \strokec4 $ modulates how each channel relates to the shared driver, and it is these inter-channel relationships that the SPIs (and hence SPI-SPI features) capture.\cb1 \
\

\f0\b \cf2 \cb3 \strokec2 ### 3.2 Temporal Misalignment via DTW and Euclidean Distance
\f1\b0 \cf4 \cb1 \strokec4 \
\

\f0\b \cf2 \cb3 \strokec2 **Setup.**
\f1\b0 \cf4 \strokec4  Channels are noisy, 
\f2\i *temporally warped*
\f1\i0  copies of the latent driver $\cf5 \strokec5 z_t\cf4 \strokec4 $. The warping for channel $\cf5 \strokec5 i\cf4 \strokec4 $ is generated by a random walk in $\cf5 \strokec5 (t_\{\cf2 \strokec2 \\\cf6 \strokec6 mathrm\cf5 \strokec5 \{orig\}\}, t_\{\cf2 \strokec2 \\\cf6 \strokec6 mathrm\cf5 \strokec5 \{warp\}\})\cf4 \strokec4 $-space: at each timestep, with probability $\cf5 \strokec5 p_\{\cf2 \strokec2 \\\cf6 \strokec6 mathrm\cf5 \strokec5 \{step\}\}^\{(i)\}\cf4 \strokec4 $, an "L-shaped" excursion deviates from the identity line. The excursion size is geometric: $\cf5 \strokec5 s \cf2 \strokec2 \\sim\cf5 \strokec5  \cf2 \strokec2 \\\cf6 \strokec6 mathrm\cf5 \strokec5 \{Geom\}(q)\cf4 \strokec4 $, with $\cf5 \strokec5 q = \cf7 \strokec7 0.5\cf4 \strokec4 $ (giving $\cf2 \strokec2 \\\cf6 \strokec6 mathbb\cf5 \strokec5 \{E\}[s] = \cf7 \strokec7 2\cf4 \strokec4 $).\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 The warp intensity parameter is $\cf5 \strokec5 p_\{\cf2 \strokec2 \\\cf6 \strokec6 mathrm\cf5 \strokec5 \{step\}\}^\{(i)\} \cf2 \strokec2 \\sim\cf5 \strokec5  \cf2 \strokec2 \\\cf6 \strokec6 mathrm\cf5 \strokec5 \{Uniform\}[\cf7 \strokec7 0\cf5 \strokec5 , a]\cf4 \strokec4 $, making $\cf5 \strokec5 a\cf4 \strokec4 $ the single control parameter governing the degree of temporal misalignment across the MTS.\cb1 \
\
\pard\pardeftab720\partightenfactor0

\f0\b \cf2 \cb3 \strokec2 **Warpedness measure.**
\f1\b0 \cf4 \strokec4  Per-channel warpedness is defined as the $\cf5 \strokec5 L_\cf7 \strokec7 1\cf4 \strokec4 $ deviation from the identity alignment:\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 $$\cf2 \strokec2 \\\cf6 \strokec6 mathcal\cf5 \strokec5 \{W\}^\{(i)\} = \cf2 \strokec2 \\\cf6 \strokec6 sum_\cf5 \strokec5 \{t=\cf7 \strokec7 1\cf5 \strokec5 \}^T |t_\{\cf2 \strokec2 \\\cf6 \strokec6 mathrm\cf5 \strokec5 \{orig\}, t\} - t_\{\cf2 \strokec2 \\\cf6 \strokec6 mathrm\cf5 \strokec5 \{warp\}, t\}^\{(i)\}|.\cf4 \strokec4 $$\cb1 \
\
\cb3 The per-MTS aggregate $\cf2 \strokec2 \\\cf6 \strokec6 mathcal\cf5 \strokec5 \{W\} = M^\{-\cf7 \strokec7 1\cf5 \strokec5 \} \cf2 \strokec2 \\sum_i\cf5 \strokec5  \cf2 \strokec2 \\\cf6 \strokec6 mathcal\cf5 \strokec5 \{W\}^\{(i)\}\cf4 \strokec4 $ satisfies $\cf2 \strokec2 \\\cf6 \strokec6 mathbb\cf5 \strokec5 \{E\}[\cf2 \strokec2 \\\cf6 \strokec6 mathcal\cf5 \strokec5 \{W\}] \cf2 \strokec2 \\approx\cf5 \strokec5  \cf7 \strokec7 2\cf5 \strokec5 aT\cf4 \strokec4 $, scaling linearly in $\cf5 \strokec5 a\cf4 \strokec4 $. The normalised form $\cf2 \strokec2 \\\cf6 \strokec6 bar\cf5 \strokec5 \{\cf2 \strokec2 \\\cf6 \strokec6 mathcal\cf5 \strokec5 \{W\}\} = (MT)^\{-\cf7 \strokec7 1\cf5 \strokec5 \} \cf2 \strokec2 \\sum_i\cf5 \strokec5  \cf2 \strokec2 \\\cf6 \strokec6 mathcal\cf5 \strokec5 \{W\}^\{(i)\}\cf4 \strokec4 $ satisfies $\cf2 \strokec2 \\\cf6 \strokec6 mathbb\cf5 \strokec5 \{E\}[\cf2 \strokec2 \\\cf6 \strokec6 bar\cf5 \strokec5 \{\cf2 \strokec2 \\\cf6 \strokec6 mathcal\cf5 \strokec5 \{W\}\}] \cf2 \strokec2 \\approx\cf5 \strokec5  \cf7 \strokec7 2\cf5 \strokec5 a\cf4 \strokec4 $, depending only on $\cf5 \strokec5 a\cf4 \strokec4 $.\cb1 \
\
\pard\pardeftab720\partightenfactor0

\f0\b \cf2 \cb3 \strokec2 **Expected behaviour of $\cf5 \strokec5 f_\{kk'\} = \cf2 \strokec2 \\\cf6 \strokec6 mathrm\cf5 \strokec5 \{corr\}(\cf2 \strokec2 \\\cf6 \strokec6 mathrm\cf5 \strokec5 \{ED\}, \cf2 \strokec2 \\\cf6 \strokec6 mathrm\cf5 \strokec5 \{DTW\})\cf2 \strokec2 $.**
\f1\b0 \cf4 \strokec4  For unwarped signals ($\cf5 \strokec5 a \cf2 \strokec2 \\approx\cf5 \strokec5  \cf7 \strokec7 0\cf4 \strokec4 $), DTW reduces to pointwise Euclidean distance (the optimal alignment path is the identity). As $\cf5 \strokec5 a\cf4 \strokec4 $ increases, DTW's flexible alignment compensates for warping while Euclidean distance degrades. The decoupling of ED and DTW across channel pairs \'97 captured by a decrease in $\cf5 \strokec5 f_\{kk'\}\cf4 \strokec4 $ \'97 is a signature of the degree of temporal misalignment in the system.\cb1 \
\

\f0\b \cf2 \cb3 \strokec2 ### 3.3 Extension Beyond Toy Examples
\f1\b0 \cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 In practice, individual features $\cf5 \strokec5 f_\{kk'\}\cf4 \strokec4 $ carry subtle signals, and it is the collective pattern across the full $\cf2 \strokec2 \\\cf6 \strokec6 binom\cf5 \strokec5 \{K\}\{\cf7 \strokec7 2\cf5 \strokec5 \}\cf4 \strokec4 $-dimensional vector that constitutes the system's dynamical signature. By selecting $\cf5 \strokec5 K\cf4 \strokec4 $ SPIs spanning diverse statistical families \'97 correlation measures, distance measures, information-theoretic quantities, spectral measures, causal/directed measures \'97 the SPI-SPI feature space forms a "meshgrid" of statistical assumptions capable of jointly characterising complex dependency regimes.\cb1 \
\
\pard\pardeftab720\partightenfactor0

\f0\b \cf2 \cb3 \strokec2 **Preliminary results.**
\f1\b0 \cf4 \strokec4  Clustering analyses in SPI-SPI space, visualised via PCA, UMAP, and t-SNE, show clear separation of synthetic MTS classes differing in dependency structure, with separation robust to variation in $\cf5 \strokec5 M\cf4 \strokec4 $ and $\cf5 \strokec5 T\cf4 \strokec4 $. Formal benchmarks against baselines are pending.\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 ---\cb1 \
\
\pard\pardeftab720\partightenfactor0

\f0\b \cf2 \cb3 \strokec2 ## 4. Open Questions and Status
\f1\b0 \cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf8 \cb3 \strokec8 -\cf4 \strokec4  The correlation choice for $\cf5 \strokec5 f_\{kk'\}\cf4 \strokec4 $ (\'a72.4) requires empirical validation.\cb1 \
\cf8 \cb3 \strokec8 -\cf4 \strokec4  The method has not been benchmarked against graph kernel baselines.\cb1 \
\cf8 \cb3 \strokec8 -\cf4 \strokec4  Real-world MTS experiments are needed.\cb1 \
\cf8 \cb3 \strokec8 -\cf4 \strokec4  The known limitation of discarding graph topology (\'a72.5) motivates the geometric deep learning extension described in the companion document.\cb1 \
\
}